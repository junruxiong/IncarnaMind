//! Native runner for the same binding, used to cross-check the wasm build:
//! csl-spike-cli <style.csl> <items.json> <citations.json> <html|text> <locale.xml>...

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!("usage: csl-spike-cli <style.csl> <items.json> <citations.json> <html|text> <locale.xml>...");
        std::process::exit(2);
    }
    let read = |p: &str| fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let locales: Vec<String> = args[5..].iter().map(|p| read(p)).collect();
    let engine = csl_spike::Engine::new(&read(&args[1]), &locales).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    match engine.render(&read(&args[2]), &read(&args[3]), &args[4], None) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
