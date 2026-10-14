// The macOS system OCR for the OCR comparison (eval/ocr/README.md): Vision's
// VNRecognizeTextRequest at its accurate level, in Simplified and Traditional
// Chinese and English, with language correction, as an app would call it.
// For each image given, prints a JSON line: its file, the milliseconds the
// request took, and the lines read, top to bottom, each with its box (Vision's
// normalised coordinates, origin at the bottom left) and confidence.
//
//   swiftc -O eval/ocr/vision.swift -o <cache>/vision-ocr && <cache>/vision-ocr a.jpg b.jpg
import Foundation
import Vision

struct Line: Codable {
  let text: String
  let confidence: Float
  let box: [Double]
}

struct Page: Codable {
  let file: String
  let milliseconds: Double
  let lines: [Line]
}

for path in CommandLine.arguments.dropFirst() {
  let url = URL(fileURLWithPath: path)
  let request = VNRecognizeTextRequest()
  request.recognitionLevel = .accurate
  request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
  request.usesLanguageCorrection = true
  let handler = VNImageRequestHandler(url: url, options: [:])
  let start = Date()
  try handler.perform([request])
  let elapsed = Date().timeIntervalSince(start) * 1000
  let lines = (request.results ?? []).compactMap { observation -> Line? in
    guard let best = observation.topCandidates(1).first else { return nil }
    let box = observation.boundingBox
    return Line(
      text: best.string, confidence: best.confidence,
      box: [box.minX, box.minY, box.maxX, box.maxY])
  }
  let data = try JSONEncoder().encode(Page(file: path, milliseconds: elapsed, lines: lines))
  print(String(data: data, encoding: .utf8) ?? "")
}
