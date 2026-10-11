//! A minimal binding around Hayagriva's CSL engine for the #288 spike:
//! a CSL style (XML), CSL locales (XML) and references (CSL JSON) in;
//! formatted citations and a bibliography out, as formatted runs (what the app would
//! use), plain text, or Hayagriva's own HTML (kept only to compare formatting).

use std::str::FromStr;

use citationberg::json as csl_json;
use citationberg::taxonomy::Locator;
use citationberg::{IndependentStyle, Locale, LocaleCode, LocaleFile, Style};
use hayagriva::{
    BibliographyDriver, BibliographyRequest, BufWriteFormat, CitationItem, CitationRequest,
    ElemChild, ElemChildren, LocatorPayload, SpecificLocator,
};
use serde::{Deserialize, Serialize};

/// One citation (an in-text cite or a footnote) of one or more references.
#[derive(Deserialize)]
pub struct CitationIn {
    pub items: Vec<CiteIn>,
    /// The footnote number; only note styles use it.
    #[serde(rename = "noteIndex")]
    pub note_index: Option<usize>,
}

#[derive(Deserialize)]
pub struct CiteIn {
    pub id: String,
    pub locator: Option<String>,
    pub label: Option<String>,
}

#[derive(Serialize)]
pub struct EntryOut {
    pub id: String,
    /// The citation number or label, when the style aligns it apart from the entry.
    #[serde(rename = "firstField")]
    pub first_field: Option<serde_json::Value>,
    pub text: serde_json::Value,
}

#[derive(Serialize)]
pub struct Output {
    pub citations: Vec<serde_json::Value>,
    pub bibliography: Vec<EntryOut>,
    #[serde(rename = "hangingIndent")]
    pub hanging_indent: bool,
}

/// A run of text with one formatting, for the "runs" output: the editor and Word export
/// build their own nodes from these, so no markup from a reference's fields is ever parsed.
#[derive(Serialize, Default)]
pub struct Run {
    pub text: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub italic: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub bold: bool,
    #[serde(rename = "smallCaps", skip_serializing_if = "std::ops::Not::not")]
    pub small_caps: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub underline: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub script: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
}

fn push_runs(children: &ElemChildren, out: &mut Vec<Run>) {
    use citationberg::{FontStyle, FontVariant, FontWeight, TextDecoration, VerticalAlign};
    let run = |f: &hayagriva::Formatted, link: Option<&String>| Run {
        text: f.text.clone(),
        italic: f.formatting.font_style == FontStyle::Italic,
        bold: f.formatting.font_weight == FontWeight::Bold,
        small_caps: f.formatting.font_variant == FontVariant::SmallCaps,
        underline: f.formatting.text_decoration == TextDecoration::Underline,
        script: match f.formatting.vertical_align {
            VerticalAlign::Sup => Some("sup"),
            VerticalAlign::Sub => Some("sub"),
            _ => None,
        },
        link: link.cloned(),
    };
    for child in &children.0 {
        match child {
            ElemChild::Text(f) => out.push(run(f, None)),
            ElemChild::Elem(e) => push_runs(&e.children, out),
            ElemChild::Link { text, url } => out.push(run(text, Some(url))),
            // Markup is Typst source and Transparent is a Typst placeholder; neither
            // arises from CSL JSON input, and neither may reach a page unescaped.
            ElemChild::Markup(m) => out.push(Run { text: m.clone(), ..Run::default() }),
            ElemChild::Transparent { .. } => {}
        }
    }
}

/// A parsed style and its locales, kept between calls so that only the first
/// call pays for parsing the XML.
pub struct Engine {
    style: IndependentStyle,
    locales: Vec<Locale>,
}

impl Engine {
    pub fn new(style_xml: &str, locales_xml: &[String]) -> Result<Engine, String> {
        let style = match Style::from_xml(style_xml).map_err(|e| format!("style: {e}"))? {
            Style::Independent(s) => s,
            Style::Dependent(_) => {
                return Err("a dependent style: pass its independent parent".into());
            }
        };
        let locales = locales_xml
            .iter()
            .map(|xml| LocaleFile::from_xml(xml).map(Locale::from))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("locale: {e}"))?;
        Ok(Engine { style, locales })
    }

    pub fn render(
        &self,
        items_json: &str,
        citations_json: &str,
        format: &str,
        locale: Option<&str>,
    ) -> Result<String, String> {
        let items: Vec<csl_json::Item> =
            serde_json::from_str(items_json).map_err(|e| format!("items: {e}"))?;
        let citations: Vec<CitationIn> =
            serde_json::from_str(citations_json).map_err(|e| format!("citations: {e}"))?;
        // "html" is Hayagriva's own writer, which does not escape field text; it is kept
        // only to compare formatting with citeproc-js. The app would use "runs".
        let format = match format {
            "html" => Some(BufWriteFormat::Html),
            "text" => Some(BufWriteFormat::Plain),
            "runs" => None,
            other => return Err(format!("unknown format {other}")),
        };
        let locale = locale.map(|l| LocaleCode(l.to_string()));

        let mut driver: BibliographyDriver<'_, csl_json::Item> = BibliographyDriver::new();
        for citation in &citations {
            let mut cites = Vec::with_capacity(citation.items.len());
            for cite in &citation.items {
                let entry = items
                    .iter()
                    .find(|item| item.id().as_deref() == Some(cite.id.as_str()))
                    .ok_or_else(|| format!("no reference with id {}", cite.id))?;
                let locator = match &cite.locator {
                    Some(text) => {
                        let label = match cite.label.as_deref() {
                            Some(label) => Locator::from_str(label)
                                .map_err(|_| format!("unknown locator label {label}"))?,
                            None => Locator::Page,
                        };
                        Some(SpecificLocator(label, LocatorPayload::Str(text)))
                    }
                    None => None,
                };
                cites.push(CitationItem::with_locator(entry, locator));
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
            driver.finish(BibliographyRequest::new(&self.style, locale.clone(), &self.locales));

        let write = |children: &ElemChildren| -> Result<serde_json::Value, String> {
            match format {
                Some(format) => {
                    let mut out = String::new();
                    children.write_buf(&mut out, format).map_err(|e| e.to_string())?;
                    Ok(serde_json::Value::String(out))
                }
                None => {
                    let mut runs = Vec::new();
                    push_runs(children, &mut runs);
                    serde_json::to_value(runs).map_err(|e| e.to_string())
                }
            }
        };

        let mut out = Output { citations: Vec::new(), bibliography: Vec::new(), hanging_indent: false };
        for citation in &rendered.citations {
            out.citations.push(write(&citation.citation)?);
        }
        if let Some(bib) = &rendered.bibliography {
            out.hanging_indent = bib.hanging_indent;
            for item in &bib.items {
                let first_field = match &item.first_field {
                    Some(field) => Some(write(&ElemChildren(vec![field.clone()]))?),
                    None => None,
                };
                out.bibliography.push(EntryOut {
                    id: item.key.clone(),
                    first_field,
                    text: write(&item.content)?,
                });
            }
        }
        serde_json::to_string(&out).map_err(|e| e.to_string())
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    /// The engine as JavaScript sees it: `new CslEngine(styleXml, [localeXml])`,
    /// then `render(itemsJson, citationsJson, "html" | "text")`, which returns JSON.
    #[wasm_bindgen]
    pub struct CslEngine(super::Engine);

    #[wasm_bindgen]
    impl CslEngine {
        #[wasm_bindgen(constructor)]
        pub fn new(style_xml: &str, locales_xml: Vec<String>) -> Result<CslEngine, JsError> {
            super::Engine::new(style_xml, &locales_xml)
                .map(CslEngine)
                .map_err(|e| JsError::new(&e))
        }

        pub fn render(
            &self,
            items_json: &str,
            citations_json: &str,
            format: &str,
            locale: Option<String>,
        ) -> Result<String, JsError> {
            self.0
                .render(items_json, citations_json, format, locale.as_deref())
                .map_err(|e| JsError::new(&e))
        }
    }
}
