//! IncarnaMind's binding to Hayagriva's CSL engine (ADR-0016), built to WebAssembly and run by
//! `src/core/citationStyles` on a worker thread.
//!
//! In: a CSL style and CSL locales as XML, then references as CSL JSON and the citations that
//! cite them. Out: the citations and the bibliography as **runs**, text with formatting flags,
//! never markup: reference metadata comes from files and lookups the app doesn't control, and
//! Hayagriva's HTML writer doesn't escape it.
//!
//! The interface is a few plain functions over the module's memory, with JSON both ways, so the
//! app loads it with a short loader of its own (`src/core/citationStyles/wasm.ts`) and the build
//! needs no wasm-bindgen:
//!
//! - `alloc(len)` and `dealloc(ptr, len)`: memory for the input.
//! - `engine_new(ptr, len)`: parses `{ style, locales }` and returns a handle, or 0 when the
//!   style can't be used; `output` then holds `{ class, title, defaultLocale }` or `{ error }`.
//! - `engine_render(handle, ptr, len)`: formats a request (`RenderIn`); returns 1, or 0 with
//!   `{ error }` in `output`.
//! - `engine_drop(handle)`.
//! - `output_ptr()` and `output_len()`: the last call's JSON.

use std::cell::RefCell;
use std::str::FromStr;

use citationberg::json as csl_json;
use citationberg::taxonomy::Locator;
use citationberg::{
    FontStyle, FontVariant, FontWeight, IndependentStyle, Locale, LocaleCode, LocaleFile,
    SecondFieldAlign, Style, StyleClass, TextDecoration, VerticalAlign,
};
use hayagriva::{
    BibliographyDriver, BibliographyRequest, CitationItem, CitationRequest, ElemChild,
    ElemChildren, ElemMeta, Formatted, LocatorPayload, SpecificLocator,
};
use serde::{Deserialize, Serialize};

/// One citation: an in-text cite or a note, of one or more references.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationIn {
    pub items: Vec<CiteIn>,
    /// The note's number; only note styles use it.
    pub note_index: Option<usize>,
}

/// One reference in a citation.
#[derive(Deserialize)]
pub struct CiteIn {
    pub id: String,
    /// Where in the work: "12", "12-13", or for a `custom` label any text ("slide 4").
    pub locator: Option<String>,
    /// A CSL locator label ("page", "line"), or "custom": the locator is printed as given, with no
    /// label. Defaults to "page".
    pub label: Option<String>,
    /// The locale this reference's terms come from ("en-US" for an English work in a Chinese
    /// style), instead of the style's.
    pub locale: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderIn {
    pub items: Vec<csl_json::Item>,
    pub citations: Vec<CitationIn>,
    /// The locale for the style's terms; none: the style's default.
    pub locale: Option<String>,
}

#[derive(Deserialize)]
pub struct EngineIn {
    pub style: String,
    pub locales: Vec<String>,
}

/// A run of text with one formatting. The app builds its own nodes from these, so no markup from a
/// reference's fields is ever parsed.
#[derive(Serialize, Default, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Run {
    pub text: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub italic: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub bold: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub small_caps: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub underline: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    /// In a citation, which of its references this run belongs to (its index in that citation's
    /// `items`, as given, before the style sorts them); none for its affixes and delimiters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<usize>,
}

impl Run {
    fn same_format(&self, other: &Run) -> bool {
        self.italic == other.italic
            && self.bold == other.bold
            && self.small_caps == other.small_caps
            && self.underline == other.underline
            && self.script == other.script
            && self.link == other.link
            && self.item == other.item
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryOut {
    pub id: String,
    /// The citation number or label, when the style sets it apart from the entry.
    pub first_field: Option<Vec<Run>>,
    pub runs: Vec<Run>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderOut {
    pub citations: Vec<Vec<Run>>,
    pub bibliography: Vec<EntryOut>,
    pub hanging_indent: bool,
    /// "flush" or "margin" when the first field (a number) is set apart; none otherwise.
    pub second_field_align: Option<&'static str>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StyleOut {
    /// "in-text" or "note".
    pub class: &'static str,
    pub title: String,
    pub default_locale: Option<String>,
    pub has_bibliography: bool,
}

fn run_of(text: &Formatted, link: Option<&String>, item: Option<usize>) -> Run {
    let f = &text.formatting;
    Run {
        text: text.text.clone(),
        italic: f.font_style == FontStyle::Italic,
        bold: f.font_weight == FontWeight::Bold,
        small_caps: f.font_variant == FontVariant::SmallCaps,
        underline: f.text_decoration == TextDecoration::Underline,
        script: match f.vertical_align {
            VerticalAlign::Sup => Some("sup"),
            VerticalAlign::Sub => Some("sub"),
            _ => None,
        },
        link: link.cloned(),
        item,
    }
}

fn push(out: &mut Vec<Run>, run: Run) {
    if run.text.is_empty() {
        return;
    }
    // Hayagriva yields many small pieces; neighbours formatted alike are one run.
    if let Some(last) = out.last_mut()
        && last.same_format(&run)
    {
        last.text.push_str(&run.text);
        return;
    }
    out.push(run);
}

fn push_runs(children: &ElemChildren, item: Option<usize>, out: &mut Vec<Run>) {
    for child in &children.0 {
        match child {
            ElemChild::Text(text) => push(out, run_of(text, None, item)),
            ElemChild::Elem(elem) => {
                let item = match elem.meta {
                    Some(ElemMeta::Entry(index)) => Some(index),
                    _ => item,
                };
                push_runs(&elem.children, item, out);
            }
            ElemChild::Link { text, url } => push(out, run_of(text, Some(url), item)),
            // Markup is Typst source and Transparent a Typst placeholder: neither arises from CSL
            // JSON, and neither may reach a page as anything but text.
            ElemChild::Markup(text) => push(out, Run { text: text.clone(), item, ..Run::default() }),
            ElemChild::Transparent { .. } => {}
        }
    }
}

fn runs(children: &ElemChildren) -> Vec<Run> {
    let mut out = Vec::new();
    push_runs(children, None, &mut out);
    out
}

/// A parsed style and its locales, kept between calls so only the first pays for parsing.
pub struct Engine {
    style: IndependentStyle,
    locales: Vec<Locale>,
}

/// Why a style can't be used.
#[derive(Serialize)]
pub struct EngineError {
    pub error: String,
    /// For a dependent style: the id of the independent style it points to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

impl Engine {
    pub fn new(style_xml: &str, locales_xml: &[String]) -> Result<Engine, EngineError> {
        let failed = |error: String| EngineError { error, parent: None };
        let style = match Style::from_xml(style_xml).map_err(|e| failed(format!("style: {e}")))? {
            Style::Independent(style) => style,
            Style::Dependent(dependent) => {
                return Err(EngineError {
                    error: "a dependent style".into(),
                    parent: Some(dependent.parent_link.href),
                });
            }
        };
        let locales = locales_xml
            .iter()
            .map(|xml| LocaleFile::from_xml(xml).map(Locale::from))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| failed(format!("locale: {e}")))?;
        Ok(Engine { style, locales })
    }

    pub fn describe(&self) -> StyleOut {
        StyleOut {
            class: match self.style.settings.class {
                StyleClass::Note => "note",
                StyleClass::InText => "in-text",
            },
            title: self.style.info.title.value.clone(),
            default_locale: self.style.default_locale.as_ref().map(|code| code.0.clone()),
            has_bibliography: self.style.bibliography.is_some(),
        }
    }

    pub fn render(&self, request: &RenderIn) -> Result<RenderOut, String> {
        let locale = request.locale.as_ref().map(|code| LocaleCode(code.clone()));
        let mut driver: BibliographyDriver<'_, csl_json::Item> = BibliographyDriver::new();
        for citation in &request.citations {
            let mut cites = Vec::with_capacity(citation.items.len());
            for cite in &citation.items {
                let entry = request
                    .items
                    .iter()
                    .find(|item| item.id().as_deref() == Some(cite.id.as_str()))
                    .ok_or_else(|| format!("no reference with id {}", cite.id))?;
                let locator = match &cite.locator {
                    Some(text) => {
                        let label = match cite.label.as_deref() {
                            None => Locator::Page,
                            Some("custom") => Locator::Custom,
                            Some(label) => Locator::from_str(label)
                                .map_err(|_| format!("unknown locator label {label}"))?,
                        };
                        Some(SpecificLocator(label, LocatorPayload::Str(text)))
                    }
                    None => None,
                };
                let item_locale = cite.locale.as_ref().map(|code| LocaleCode(code.clone()));
                cites.push(CitationItem::new(entry, locator, item_locale, false, None));
            }
            driver.citation(CitationRequest::new(
                cites,
                &self.style,
                locale.clone(),
                &self.locales,
                citation.note_index,
            ));
        }

        let rendered =
            driver.finish(BibliographyRequest::new(&self.style, locale, &self.locales));
        let citations = rendered.citations.iter().map(|each| runs(&each.citation)).collect();
        let mut out = RenderOut {
            citations,
            bibliography: Vec::new(),
            hanging_indent: false,
            second_field_align: None,
        };
        if let Some(bibliography) = &rendered.bibliography {
            out.hanging_indent = bibliography.hanging_indent;
            out.second_field_align = bibliography.second_field_align.map(|align| match align {
                SecondFieldAlign::Flush => "flush",
                SecondFieldAlign::Margin => "margin",
            });
            for item in &bibliography.items {
                out.bibliography.push(EntryOut {
                    id: item.key.clone(),
                    first_field: item
                        .first_field
                        .as_ref()
                        .map(|field| runs(&ElemChildren(vec![field.clone()]))),
                    runs: runs(&item.content),
                });
            }
        }
        Ok(out)
    }
}

thread_local! {
    /// The engines made, by handle - 1; a dropped one leaves None.
    static ENGINES: RefCell<Vec<Option<Engine>>> = const { RefCell::new(Vec::new()) };
    /// The last call's JSON, read through `output_ptr` and `output_len`.
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_output<T: Serialize>(value: &T) {
    let json = serde_json::to_vec(value)
        .unwrap_or_else(|_| br#"{"error":"the output couldn't be written"}"#.to_vec());
    OUTPUT.with(|output| *output.borrow_mut() = json);
}

fn error(message: impl Into<String>) {
    set_output(&EngineError { error: message.into(), parent: None });
}

/// The input the caller wrote at `ptr`, as text.
///
/// # Safety
/// `ptr` and `len` must describe memory the caller filled after `alloc(len)`.
unsafe fn input<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).map_err(|e| format!("input: {e}"))
}

/// Memory for the caller to write `len` bytes of input into.
#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Frees what `alloc(len)` gave.
///
/// # Safety
/// `ptr` must come from `alloc(len)` with the same `len`, and be freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

/// Parses `{ style, locales }`: a handle (from 1), or 0 with the reason in the output.
///
/// # Safety
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn engine_new(ptr: *const u8, len: usize) -> u32 {
    let parsed = unsafe { input(ptr, len) }
        .and_then(|text| serde_json::from_str::<EngineIn>(text).map_err(|e| format!("input: {e}")));
    let request = match parsed {
        Ok(request) => request,
        Err(message) => {
            error(message);
            return 0;
        }
    };
    match Engine::new(&request.style, &request.locales) {
        Ok(engine) => {
            set_output(&engine.describe());
            ENGINES.with(|engines| {
                let mut engines = engines.borrow_mut();
                engines.push(Some(engine));
                engines.len() as u32
            })
        }
        Err(failure) => {
            set_output(&failure);
            0
        }
    }
}

/// Formats a request with an engine: 1, or 0 with the reason in the output.
///
/// # Safety
/// As `input`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn engine_render(handle: u32, ptr: *const u8, len: usize) -> u32 {
    let parsed = unsafe { input(ptr, len) }
        .and_then(|text| serde_json::from_str::<RenderIn>(text).map_err(|e| format!("input: {e}")));
    let request = match parsed {
        Ok(request) => request,
        Err(message) => {
            error(message);
            return 0;
        }
    };
    let result = ENGINES.with(|engines| {
        let engines = engines.borrow();
        match handle.checked_sub(1).and_then(|index| engines.get(index as usize)) {
            Some(Some(engine)) => engine.render(&request),
            _ => Err(format!("no engine {handle}")),
        }
    });
    match result {
        Ok(out) => {
            set_output(&out);
            1
        }
        Err(message) => {
            error(message);
            0
        }
    }
}

/// Frees an engine.
#[unsafe(no_mangle)]
pub extern "C" fn engine_drop(handle: u32) {
    ENGINES.with(|engines| {
        let mut engines = engines.borrow_mut();
        if let Some(slot) = handle.checked_sub(1).and_then(|index| engines.get_mut(index as usize))
        {
            *slot = None;
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn output_ptr() -> *const u8 {
    OUTPUT.with(|output| output.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn output_len() -> usize {
    OUTPUT.with(|output| output.borrow().len())
}
