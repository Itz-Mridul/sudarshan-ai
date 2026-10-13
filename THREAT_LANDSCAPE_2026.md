# Steganography Threat Landscape — 2026

This note separates observed tradecraft from research directions. It is a
threat-modeling aid for Sudarshana, not proof that every technique is used in
every campaign.

## What the detector should expect

- **Pixel and frequency hiding:** LSB, adaptive spatial schemes, and JPEG/DCT
  embedding remain relevant. These require algorithm-diverse training and
  cover-source testing; BOSSBase-only results do not establish field
  performance.
- **Container and parser abuse:** payloads may be appended after a valid image
  boundary, hidden in SVG/HTML comments, or reconstructed by a loader in
  memory. The current project therefore audits PNG IEND, JPEG EOI, GIF, BMP,
  and RIFF/WebP boundaries and preserves trailing bytes for offline analysis.
- **Multi-stage delivery:** recent reporting describes steganographic loaders
  that extract a payload from an image and continue in memory. Fortinet's
  2026 PawsRunner report and Elastic's 2026 Contagious Interview report are
  useful examples of why a file-boundary audit cannot replace pixel analysis.
- **AI-generated and multimodal carriers:** generative steganography,
  diffusion carriers, and text/image payload research are important future
  threat models. They must be treated as hypotheses until the detector has
  labelled training and cross-domain test data; a classical CNN score is not
  evidence that it detects latent or multimodal payloads.

## Verified 2025–2026 examples to track

- [Fortinet: PureLogs delivery via PawsRunner steganography](https://www.fortinet.com/lat/blog/threat-research/purelogs-delivery-via-pawsrunner-steganography)
  describes a 2026 campaign using steganography in a multi-stage loader.
- [Elastic Security Labs: Contagious Interview SVG steganography](https://security-labs.elastic.co/security-labs/contagious-interview-malware-svg-steganography)
  reports malicious code split across SVG image assets and reassembled by a
  JavaScript component.
- [Acronis: FileFix in the wild](https://www.acronis.com/en/tru/posts/filefix-in-the-wild-new-filefix-campaign-goes-beyond-poc-and-leverages-steganography/)
  documents 2025 reporting on scripts and encrypted payloads concealed in
  apparently ordinary images.
- [Stego Battlefield / SADBench](https://arxiv.org/abs/2605.05789) is a 2026
  research direction that evaluates attack capability, detection, efficiency,
  and transferability across image and text payloads.

## Claims that should not be stated as established facts

The project should not present “reversible generative steganography,”
multimodal prompt-injection carriers, Magecart EXIF skimming, or a named
“StoatWaffle steganography campaign” as operational facts without a primary
incident report and a reproducible sample. They are useful hypotheses for a
future test plan, not current evidence for this detector.

## Response pipeline

1. Validate magic bytes and the format-specific structural boundary.
2. Carve trailing bytes to a quarantined artifact; never execute or decode it
   automatically.
3. Run RS and chi-square as source-calibrated statistical screens.
4. Run the learned multi-branch model only within its validated training
   scope.
5. Preserve the original hash, parameters, model version, and analyst notes.
