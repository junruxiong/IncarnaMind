//! Native timing of the same binding, to separate engine cost from wasm cost:
//! csl-spike-bench <style.csl> <items.json> <citations.json> <runs> <locale.xml>...

use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let read = |p: &str| fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let runs: usize = args[4].parse().expect("runs");
    let locales: Vec<String> = args[5..].iter().map(|p| read(p)).collect();
    let (style, items, cites) = (read(&args[1]), read(&args[2]), read(&args[3]));

    let t = Instant::now();
    let engine = csl_spike::Engine::new(&style, &locales).expect("engine");
    let parse = t.elapsed();
    let t = Instant::now();
    engine.render(&items, &cites, "runs", None).expect("render");
    let first = t.elapsed();
    let mut warm = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        engine.render(&items, &cites, "runs", None).expect("render");
        warm.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{{\"parseMs\":{:.2},\"firstRenderMs\":{:.2},\"warmRenderMedianMs\":{:.2}}}",
        parse.as_secs_f64() * 1000.0,
        first.as_secs_f64() * 1000.0,
        warm[warm.len() / 2]
    );
}
