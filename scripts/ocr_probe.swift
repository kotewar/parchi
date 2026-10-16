// Proves the macOS Vision OCR path end-to-end, with no model and no network.
//
// Renders a synthetic prescription in a HANDWRITING font, then reads it back
// with VNRecognizeTextRequest. This is the layer that replaces a vision model:
// the community benchmark measured a dedicated engine at 0.27s/image versus
// 82.8s for a local 27B vision model, with 4x fewer character errors.
//
// Build & run:
//   swiftc -O scripts/ocr_probe.swift -o /tmp/ocr_probe && /tmp/ocr_probe

import AppKit
import Foundation
import Vision

let prescription = """
Dr. R. Deshmukh   MBBS   Reg 2019/4482
Pt: Mr. Suresh Wankhede   42/M
Dx: HTN, T2DM

Rx
1. Tab Telma 40mg        1-0-0    x 30 days
2. Tab Glycomet GP2      1-0-1    x 30 days
3. Cap Pan 40            1-0-0    x 15 days
4. Inj. Insulin          0-0-1    x 10 days
"""

let width = 1100.0
let height = 620.0

let image = NSImage(size: NSSize(width: width, height: height))
image.lockFocus()
NSColor.white.setFill()
NSRect(x: 0, y: 0, width: width, height: height).fill()

// A handwriting face if the machine has one, else something legible.
let handwritingNames = ["Noteworthy-Light", "BradleyHandITCTT-Bold", "Chalkboard", "MarkerFelt-Thin"]
var chosenFont = NSFont.systemFont(ofSize: 30)
var chosenName = "system (no handwriting font found)"
for name in handwritingNames {
    if let f = NSFont(name: name, size: 30) {
        chosenFont = f
        chosenName = name
        break
    }
}
print("font used: \(chosenName)")

let attrs: [NSAttributedString.Key: Any] = [
    .font: chosenFont,
    .foregroundColor: NSColor.black,
]
prescription.draw(
    in: NSRect(x: 40, y: 40, width: width - 80, height: height - 80),
    withAttributes: attrs
)
image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let png = rep.representation(using: .png, properties: [:]) else {
    print("FATAL: could not render test image")
    exit(1)
}
let outURL = URL(fileURLWithPath: "/tmp/rx_test.png")
try png.write(to: outURL)
print("rendered -> \(outURL.path)")

guard let cgImage = NSImage(contentsOf: outURL)?
    .cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("FATAL: could not reload rendered image")
    exit(1)
}

// --- the OCR layer ---
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false  // drug names are not English words

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

let t0 = Date()
do {
    try handler.perform([request])
} catch {
    print("FATAL: Vision request failed: \(error)")
    exit(1)
}
let elapsed = Date().timeIntervalSince(t0)

let observations = request.results ?? []

print(String(format: "OCR time: %.3fs for %d text observations", elapsed, observations.count))
print("--- recognised text ---")
for obs in observations {
    if let candidate = obs.topCandidates(1).first {
        let box = obs.boundingBox
        print(String(format: "[%.2f conf, y=%.2f] %@",
                     candidate.confidence, box.origin.y, candidate.string))
    }
}

// Vision returns no per-document confidence, so aggregation is ours to do.
if !observations.isEmpty {
    let confidences = observations.compactMap { $0.topCandidates(1).first?.confidence }
    let mean = confidences.reduce(0, +) / Float(confidences.count)
    let low = confidences.filter { $0 < 0.5 }.count
    print(String(format: "mean recognition confidence: %.3f | low-confidence lines: %d/%d",
                 mean, low, confidences.count))
}
