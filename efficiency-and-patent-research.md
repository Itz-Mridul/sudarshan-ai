# MBCSS — EFFICIENCY & PATENT-READINESS RESEARCH ADDENDUM
### Deep-Research Findings from Literature, Code Ecosystem, Patent Databases & India Patent Office (2025–2026)
**Companion to `prd.md` | Researched: September 2026**

---

## HOW TO USE THIS DOCUMENT

This file does not replace `prd.md` — it sits next to it. Everything below was found by
researching the live literature, the PyTorch/ONNX ecosystem, and patent databases (Google
Patents, USPTO, Indian Patent Office guidance) **after** your document was written, specifically
to answer two questions: *(1) can MBCSS be made faster/smaller/more accurate, and (2) will its
four patent claims actually survive examination.*

Sections marked **🔴 HIGH PRIORITY** describe things that could break your current plan if left
unaddressed — read those first. Everything here is additive: keep using `prd.md` as your
week-by-week driver, and pull fixes from here as you hit the relevant milestone.

**Legal note:** Part 2 is general information assembled from public sources, not legal advice.
Nothing here is a substitute for a registered Indian patent agent, who must review and finalize
any claim language before filing — this is stated in `prd.md` too, and it's worth repeating.

---

# PART 1 — MAKING MBCSS MORE EFFICIENT

## 1.1 🔴 Your quantization code targets an API PyTorch is actively deleting

`prd.md` §4.7 calls `torch.quantization.get_default_qconfig('fbgemm')`,
`torch.quantization.prepare`, and `torch.quantization.convert`. This is PyTorch's **eager-mode
quantization** API. As of PyTorch's current documentation, eager-mode and FX-graph-mode
quantization are both marked for migration to a new library, **torchao**, with the old
`torch.ao.quantization` namespace scheduled for deletion once blockers clear. The functions
still work today, but writing new project code against them in September 2026 means building on
an API in its final months.

**What replaces it:** `torchao`'s **PT2E** flow (`prepare_pt2e` / `convert_pt2e`), built on
`torch.export`. Sketch of the equivalent pipeline for your fusion model:

```python
# pip install torchao
import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer

model.eval()
example_inputs = (pixel_map, dct_map, stat_feat)     # your 3-branch forward signature
exported = export(model, example_inputs).module()

quantizer = X86InductorQuantizer()                    # targets x86 CPU — matches your laptop
prepared = prepare_pt2e(exported, quantizer)

with torch.no_grad():                                 # same calibration idea as before
    for batch in calibration_loader:
        prepared(*batch)

quantized_model = convert_pt2e(prepared)
```

This API is still evolving (it moved out of "prototype" status only recently), and multi-input
forward signatures like yours are less battle-tested than single-input CNN examples in the
docs. **Practical recommendation:** try PT2E first since it's the forward-compatible path; if you
hit export errors on the 3-branch forward pass, FX-graph-mode quantization (`quantize_fx`) is
still functional as a fallback and is much better documented for multi-branch models than PT2E
currently is. Either way, avoid writing new tutorials/report text that assumes the eager-mode
API will still be current when your paper is reviewed in 2027.

## 1.2 QAT vs. PTQ — the accuracy drop can be far worse than "1–5%"

`prd.md` budgets a 1–5% accuracy drop from INT8 quantization and uses **post-training
quantization (PTQ)** — calibrate on 100–200 samples, convert, done. The literature shows this
assumption doesn't hold for all architectures. NVIDIA's own quantization benchmarks report
EfficientNet-B0 dropping from a 77.4% FP32 baseline to **33.9%** accuracy under PTQ — but
recovering to 76.8% (nearly lossless) when the *same* model is quantized with
**quantization-aware training (QAT)** instead. ResNet-style architectures are typically far more
PTQ-tolerant than EfficientNet-style ones (depthwise-separable and squeeze-excite layers are
known to be more quantization-sensitive), so your actual drop depends heavily on which backbone
you end up using per §1.7 below.

**Recommendation:** don't assume PTQ will land you in the 1–5% band — measure it, and if it
doesn't, insert a short QAT phase (fake-quantize nodes inserted during the last few fine-tuning
epochs from RULE-TRAIN, §6.2) before falling back to FP16 per your own Risk Register mitigation.
This also gives you a genuinely interesting number for your paper's ablation table: *"PTQ vs.
QAT accuracy retention"* is a publishable comparison in its own right, not just an engineering
footnote.

## 1.3 Beyond PyTorch: dedicated CPU-inference runtimes

PyTorch's built-in `fbgemm` backend is a reasonable baseline, but the entire CPU-inference
ecosystem has consolidated around a few tools specifically because raw framework quantization
tends to leave performance on the table:

| Tool | What it adds over raw PyTorch quantization |
|------|---------------------------------------------|
| **ONNX Runtime** | Export once, get a graph-optimized runtime with its own INT8 quantization pass and operator fusion; portable off PyTorch entirely for the demo/deploy stage |
| **Intel Neural Compressor** | Accuracy-driven *automatic* quantization tuning — searches per-layer precision to hit a target accuracy rather than quantizing everything uniformly; also bundles pruning and distillation in one tool |
| **OpenVINO (NNCF)** | Purpose-built for Intel-CPU inference (i.e., exactly "any CPU laptop"); supports both PTQ and QAT through the same Neural Network Compression Framework, with documented INT8/FP32 comparison benchmarks per network |

None of these require abandoning your PyTorch training pipeline — you'd train as planned, then
export the trained (or QAT-fine-tuned) model through ONNX into whichever runtime gives the best
size/latency numbers on your actual laptop. This turns your NFR-01 (<500ms) and NFR-02 (<20MB)
targets into a runtime *bake-off* you can put directly in the paper: "PyTorch fbgemm vs. ONNX
Runtime vs. OpenVINO on identical hardware" is a table reviewers like, because it's a controlled,
reproducible comparison rather than a single anecdotal number.

## 1.4 Branch B's DCT computation has two separable problems

Looking at your `compute_dct_blocks()` (§4.3): it loops over every 8×8 block in Python and calls
`scipy.fft.dct` twice per block. For a 512×512 image that's 4,096 blocks × 10,000 images ≈ 41
million individual DCT calls during dataset preparation alone, almost entirely spent on Python
loop and function-call overhead rather than actual math.

**Fix 1 — vectorize it (applies regardless of format).** Reshape the image into a block tensor
and run `scipy.fft.dctn` once across all blocks simultaneously:

```python
import numpy as np
import scipy.fft as fft

def compute_dct_blocks_fast(image_array, block_size=8):
    """Same output as your nested-loop version, one scipy call instead of ~4,096."""
    H, W = image_array.shape
    Hc, Wc = H - H % block_size, W - W % block_size
    img = image_array[:Hc, :Wc].astype(np.float32)

    blocks = img.reshape(Hc // block_size, block_size, Wc // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)                       # (nH, nW, 8, 8)
    dct_blocks = fft.dctn(blocks, axes=(2, 3), norm='ortho')    # all blocks at once
    return dct_blocks.transpose(0, 2, 1, 3).reshape(Hc, Wc)
```

**Fix 2 — for your J-UNIWARD test set specifically, don't recompute the DCT at all.** J-UNIWARD
embeds directly into the *quantized* DCT coefficients JPEG already stores on disk. Recomputing a
float DCT from the decoded pixels (as your current code does) is an approximation of the actual
embedding domain, not the domain itself — the embedding-induced perturbation is smaller than the
rounding difference between "true quantized coefficient" and "recomputed float coefficient" at
high JPEG quality. Two maintained Python packages expose the real coefficients directly from the
JPEG bitstream via libjpeg: **`jpegio`** and **`jpeglib`**.

```python
import jpegio as jio

def get_real_jpeg_dct(jpeg_path):
    """Reads the ACTUAL quantized DCT coefficients JPEG stored, instead of
    re-deriving an approximate float DCT from decoded pixels. Use this for
    J-UNIWARD (and any other JPEG-domain stego) — BOSSBase PGM images have
    no JPEG container, so Fix 1's vectorized recompute still applies there."""
    jpeg = jio.read(jpeg_path)
    y_dct = jpeg.coef_arrays[0]         # luminance channel, quantized DCT
    quant_table = jpeg.quant_tables[0]  # the quant matrix actually used
    return y_dct, quant_table
```

This is also a stronger methodology claim for your paper: "Branch B operates on the JPEG
codec's native coefficients" is a more defensible sentence in front of a reviewer than "Branch B
recomputes an approximate DCT."

## 1.5 A 2026 finding that complicates RULE-ARCH-02

RULE-ARCH-02 freezes all 30 SRM filters permanently, reasoning that gradient flow through them
causes instability. A March 2026 study on lightweight steganalysis CNNs (testing
MobileNetV2/EfficientNetV2S with SRM-style high-pass preprocessing) found the opposite in one
specific respect: allowing the high-pass filters to adapt during training — starting from the
Fridrich 2012 initialization rather than random init, but *not* freezing them — produced an
additional accuracy gain over keeping them permanently fixed, because the filters could adjust
to the statistical character of the specific stego noise being detected.

This doesn't necessarily contradict your rule so much as sharpen it: the instability RULE-ARCH-02
is protecting against likely comes from unfreezing SRM filters *from the start of training on a
randomly-initialized network*, not from ever unfreezing them at all. Your own RULE-ARCH-03
already unfreezes "everything" during the joint fine-tuning stage at a very low learning rate
(1e-5) — which is precisely the regime (pretrained network, tiny LR) where adapting the SRM
filters is least likely to destabilize training. **Concrete suggestion:** in Week 10's joint
fine-tuning pass, try simply removing `requires_grad = False` from the SRM layer alongside
everything else you're already unfreezing, and report whether it helps — you get a free ablation
row ("frozen SRM vs. fine-tuned SRM") without changing your training schedule at all.

## 1.6 Fusion layer: a learned gate is a small, well-precedented upgrade over fixed concatenation

Your LayerNorm-then-concatenate fusion (§4.5, RULE-ARCH-01) correctly solves the magnitude
imbalance problem (256-dim vs. 64-dim branches). A step beyond fixed concatenation, well
established in the multi-branch fusion literature (Squeeze-and-Excitation-style channel
attention has been repeatedly shown, across domains from EEG classification to hyperspectral
imaging, to outperform plain concatenation for multi-branch fusion at very low parameter cost)
is to let the network learn *per-sample* how much to weight each branch, rather than always
weighting them equally after normalization. Applied to your specific 3-branch setup:

```python
class SEGatedFusion(nn.Module):
    """Small, per-sample-adaptive alternative to fixed concatenation. Adds a
    single tiny gating MLP; everything downstream (fc1→fc2→fc3) is unchanged."""
    def __init__(self, dim_a=256, dim_b=256, dim_c=64, reduction=8):
        super().__init__()
        total = dim_a + dim_b + dim_c
        self.norm_a, self.norm_b, self.norm_c = (
            nn.LayerNorm(dim_a), nn.LayerNorm(dim_b), nn.LayerNorm(dim_c)
        )
        self.gate = nn.Sequential(
            nn.Linear(total, total // reduction), nn.ReLU(),
            nn.Linear(total // reduction, 3), nn.Softmax(dim=1)
        )

    def forward(self, fa, fb, fc):
        fa, fb, fc = self.norm_a(fa), self.norm_b(fb), self.norm_c(fc)
        w = self.gate(torch.cat([fa, fb, fc], dim=1))          # [batch, 3]
        return torch.cat([fa * w[:, 0:1], fb * w[:, 1:2], fc * w[:, 2:3]], dim=1)
```

This is a genuine architectural contribution you can ablate ("fixed concat vs. learned gate") —
and, separately, it's also relevant to Part 2: a *learned, confidence-adaptive* fusion mechanism
is a more specific, more defensible thing to claim in a patent than "a fusion layer," which (per
§2.1) is a heavily occupied area.

## 1.7 Lightweight backbones worth benchmarking against your custom branches

You're building Branch A and Branch B from scratch. Before committing months of tuning to
custom architectures, it's worth knowing where the field has already landed on the
accuracy/compute trade-off, since several of these are drop-in transfer-learning options that
could save you weeks:

| Approach | Reported result | Why it's relevant to you |
|---|---|---|
| **EfficientNet-B0 + block removal** (Scientific Reports, 2023) | 90.73% (BOSSBase), 82.40% (ALASKA2); 9.58× fewer parameters, 2.16× fewer FLOPs than baseline | Shows most of the accuracy in steganalysis CNNs comes from a *few early layers* — directly actionable if Branch A ends up too slow |
| **MobileNetV2 / EfficientNetV2S + HPF** (Mar 2026) | High accuracy on LSB at multiple payloads with much lower compute than specialized nets | Closest published analogue to what Branch A is trying to do — worth citing as a baseline you're competitive with |
| **StegResNet** (ResNet18, ImageNet-pretrained, partially frozen) | 91.27% combined BOSSBase+BOWS2, outperforming XuNet/YeNet/YedroudjNet/SRNet/ZhuNet/GBRASNet | Transfer learning from ImageNet weights, not trained from scratch — plausibly the fastest path to a strong Branch A given your Colab-time constraints |
| **Green Steganalyzer (GS)** | Comparable detection to deep-learning SOTA against S-UNIWARD/WOW/HILL, with much lower computational complexity and smaller model size | Not a CNN at all — uses Saab transforms + self-supervised anomaly scoring, no backpropagation. Worth a mention in your literature review as evidence that *non-gradient-based* approaches are a live alternative, and a strong napkin comparison for "why did you choose deep learning" in your viva |
| **PENet+** (arXiv:2606.10939, already in your doc) | 45.5% fewer parameters via HPF-redundancy reduction + inverted-residual backbone | Already cited; the paper's specific technique (reducing *redundant* high-pass filter responses rather than removing filters outright) is a finer-grained idea than plain pruning if you want to keep all 30 SRM filters but cut their downstream compute cost |

None of these require abandoning your 3-branch design — they're candidate implementations *for*
Branch A/B, evaluated against your existing accuracy targets (FR-01/FR-02) before you commit to a
from-scratch architecture for the full 3-month build.

## 1.8 Two more compression paths to run alongside INT8 quantization

- **Knowledge distillation.** Train (or fine-tune) a larger, unconstrained fusion model first
  as a "teacher," then distill it into a smaller "student" using the teacher's softened output
  probabilities as an extra training signal. This is a different lever from quantization (it
  changes *what the network learns*, not how its weights are stored), so the two combine — the
  literature on edge deployment routinely applies distillation *and* quantization to the same
  model. If your Branch A/B/C-then-fusion pipeline already trains branches separately before
  joint fine-tuning (RULE-ARCH-03), you already have a natural "teacher" (branches unfrozen) and
  candidate "student" (a slimmer fused head) split without restructuring your schedule.
- **Structured pruning**, distinct from quantization: removing whole channels/filters (not just
  representing existing ones in fewer bits). Intel Neural Compressor (§1.3) bundles pruning,
  quantization, and distillation behind one API, which is worth knowing if you want to combine
  all three without hand-rolling each separately.

## 1.9 A concrete, citable answer to your own Risk Register / research-gap claim

`prd.md` cites Frontiers AI 2025 to justify the claim that existing detectors "degrade under
compression/resize." A follow-up robustness study (Frontiers in AI, evaluating EfficientNet,
SRNet, ResNet, Xu-Net, and Yedroudj-Net under resizing, compression, cropping, and noise on
BOSSBase) makes this concrete: **EfficientNet-based architectures were the most robust of the
five to these transformations**, while Xu-Net and Yedroudj-Net — your two named baselines —
showed the sharpest degradation, especially under added noise. Two actionable takeaways:

1. This is a stronger, more specific citation for your "research gap" paragraph than the generic
   ACM Computing Surveys line — you can now name *which* architecture family holds up and *which
   transformations* hurt most, which reviewers will find more convincing than a general
   assertion.
2. If robustness-to-transformation becomes part of your contribution story, training with
   augmentation that includes mild JPEG re-compression and resizing (not just the horizontal
   flip your RULE-TRAIN-08 currently allows) is a low-cost way to close some of this gap — the
   same source recommends transformation-diverse training and domain adaptation as the practical
   mitigation. Note this only affects the LSB/spatial pathway's augmentation; it doesn't
   contradict RULE-TRAIN-08's reasoning about rotation/crop destroying stego signal, since resize
   and re-compression are different operations from geometric cropping.

## 1.10 Efficiency recommendations, ranked by effort vs. payoff

| Priority | Change | Effort | Payoff |
|---|---|---|---|
| 1 | Vectorize `compute_dct_blocks` (§1.4, Fix 1) | Minutes | Materially faster dataset prep, zero accuracy risk |
| 2 | Measure actual PTQ drop before assuming 1–5%; add QAT if it's worse (§1.2) | Low | Avoids an unpleasant surprise in Week 11 |
| 3 | Use `jpegio`/`jpeglib` for the J-UNIWARD test path (§1.4, Fix 2) | Low | More correct methodology, stronger paper claim |
| 4 | Try unfreezing SRM filters during the existing Week-10 fine-tune (§1.5) | Low (one flag) | Possible free accuracy gain + a free ablation row |
| 5 | Benchmark ONNX Runtime / OpenVINO against raw PyTorch fbgemm (§1.3) | Medium | Directly targets NFR-01/NFR-02; good paper table |
| 6 | Evaluate a transfer-learned backbone (StegResNet-style) for Branch A (§1.7) | Medium | Could shortcut weeks of from-scratch tuning |
| 7 | Swap fixed concat for the SE-gated fusion (§1.6) | Medium | Ablatable contribution + strengthens Claim 1's novelty (see Part 2) |
| 8 | Migrate quantization code off the deprecated eager-mode API (§1.1) | Medium | Future-proofs the codebase past PyTorch's removal timeline |
| 9 | Add mild resize/recompression augmentation (§1.9) | Medium | Targets your own stated research gap directly |
| 10 | Knowledge distillation / structured pruning as a second compression axis (§1.8) | High | Extra ablation material; not required to hit your accuracy targets |

---

# PART 2 — MAKING MBCSS PATENT-READY

## 2.1 Prior-art check: what already exists for each of your four claims

The Risk Register in `prd.md` already flags "patent invalidated by prior art" as a real risk and
correctly notes the mitigation is to claim the *combination*, not the individual pieces. Here is
what a prior-art search actually turns up for each piece — this substantiates that mitigation
with specifics rather than a general worry.

**Claim 1 (3-branch fusion → confidence-scored output).** The general pattern of *multiple
neural-network branches processing different feature representations of the same input, fused by
a downstream layer into a classification* is well-established prior art at the architecture
level, independent of steganalysis:

- **US 10,068,171 B2** / US20170140253A1 (Xerox/Conduent) — multi-layer fusion in a CNN for
  image classification, fusing features from different representations of an input image.
- **US 11,556,784 B2** (Samsung) — a multi-task fusion neural network architecture combining
  multiple branch outputs.
- **CN 109063765B** — an image classification method built around *gated* neural-network
  information fusion, explicitly framed around multi-branch fusion being a widespread pattern
  (citing ResNet, DenseNet, GRU as prior examples of branch-fusion operations).

None of these are steganalysis-specific, and none use your specific pixel-residual /
DCT-coefficient / statistical-feature triple. That's exactly why your Risk Register's instinct
to claim the *combination* is correct — but it also means Claim 1, as currently drafted
("comprising (a)...(b)...(c)...(d)..."), reads structurally identical to prior art multi-branch
fusion claims with the *domain of the three branches* swapped in. Under India's technical-effect
doctrine (§2.3), that can still be patentable, but the specification needs to argue the technical
effect explicitly — measurable accuracy gain from the *specific* fusion of these three particular
domains for *this* particular problem — rather than relying on the architecture shape alone.

**Claim 3 (INT8 CPU deployment).** INT8 quantization for edge/CPU deployment is one of the most
heavily patented areas in applied ML — Qualcomm, Google, ARM, Intel, and NVIDIA all hold
extensive portfolios here, and the *general technique* (quantize a trained network to INT8,
calibrate, deploy on CPU) has essentially no remaining novelty on its own. This claim only has
force as a *dependent* claim tied to the specific fusion architecture — "the system of Claim 1,
deployed via INT8 quantization" — never as a standalone method claim over quantization in the
abstract, which prior art overwhelmingly forecloses.

**Claim 4 (method for detecting insider-threat exfiltration via steganographic images in
air-gapped networks).** The Forensic Focus literature confirms that connecting steganography
detection to the insider-threat problem is itself a long-recognized idea, not a new one — the
framing goes back over a decade in the digital-forensics community. The novelty here can only be
in the *specific technical method* (your architecture + deployment + scoring combination), not
in the observation that steganography enables exfiltration, which is prior art as a problem
statement.

## 2.2 🔴 Claim 2 (defense-context scoring) is your weakest claim — here's why

This is the most important finding in this document. Your `compute_risk_score()` function
combines a model confidence score with weighted additions for user privilege, time-of-day
anomaly, file-size anomaly, and destination type into a single 0–100 score with four severity
tiers. That exact pattern — combine behavioral/contextual signals into a weighted composite risk
score with severity thresholds — is the foundational mechanism of the entire commercial
insider-threat/UEBA industry, and it is both extensively product-shipped and extensively
patented:

- **US 11,611,574 B2** / **US 12,137,110 B2** ("User behavior analytics for insider threat
  detection") describe computing a risk score from weighted signal combinations against a
  threshold, with weights that can be role- or document-type-specific and can be machine-learned
  rather than hand-set — structurally the same mechanism as your privilege/time/size/destination
  formula, just with different named signals.
- Commercial platforms (Exabeam, Proofpoint, Teramind, Fidelis, and others) all describe
  combining user privilege, behavioral anomaly severity, and data sensitivity into "a single risk
  score" as their core, publicly marketed mechanism — this is now industry-standard practice,
  not a novel combination.

**What this means concretely:** Claim 2 as written — "a defense-context scoring module
integrating: user privilege, time of transfer, file size anomaly ratio, and destination network
address into a composite risk score" — describes the *category* of thing every DLP/UEBA vendor
already sells. An examiner (or an opposing party, later) can defeat this claim with any of the
above as prior art in about the time it takes to run one search.

**How to fix it:** the genuinely novel element you have is not "we compute a weighted risk
score" — it's that **one of the score's inputs is a live, multi-domain CNN steganalysis
confidence value**, not a behavioral-log signal. Every citation above scores *behavior and
metadata*; none of them fuse in a *learned image-content classifier's* output as a first-class,
architecturally-integrated signal. Recommend redrafting Claim 2 to be explicit and narrow about
this: the risk score module receives its stego-confidence input specifically from the fused
output of Claim 1's three-branch network (not a generic malware/DLP scanner score), and the
patent's technical-effect argument should rest on that architectural coupling — "steganalysis
confidence as a structural input, not a bolt-on" — rather than on the scoring formula, which
is not new.

## 2.3 CRI Guidelines 2025 — confirmed accurate, with additional depth worth knowing

`prd.md`'s claim that the Indian Patent Office issued new CRI Guidelines effective July 29, 2025
is accurate and current as of this research (confirmed via multiple independent legal-analysis
sources as recently as March 2026, with no indication of the guidelines being withdrawn or
successfully challenged since). Details worth adding to your specification-drafting process:

- **The guidelines replace the 2017 version** and were finalized after two public draft rounds
  (March and June 2025) plus stakeholder consultations in multiple cities — they're the product
  of an unusually open process, which Indian patent commentary treats as making them more durable
  than a typical circular.
- **The core test is "technical effect."** A claim must demonstrate a concrete, measurable
  improvement to a technical system — faster processing, reduced computing resources, real-time
  operational gains — not just an abstract algorithm dressed in system language. This is
  precisely why your Risk Register's "technical effect" bullets (accuracy improvement, 4×
  smaller model, CPU-only deployment) are the right things to be emphasizing — keep them front
  and center in the specification's problem/solution framing.
- **The "Seven Stambhas" (pillars) novelty framework** is the new guidelines' structured approach
  to novelty assessment — a step-by-step method examiners now use to evaluate what a claim
  actually covers before deciding patentability, replacing more ad hoc 2017-era assessment. A
  patent agent drafting your claims should structure the specification's "advantages" section to
  map cleanly onto this framework rather than assuming a general "our system is better" narrative
  will suffice.
- **A two-step test now governs the algorithm exclusion** (Section 3(k)): first, examine the
  actual substance of the claim past its wording; second, decide whether that substance is
  abstract or concretely implemented. A claim describing your architecture only in the abstract
  ("a method comprising extracting features and classifying them") risks failing step two; a
  claim tied to concrete implementation detail (specific branch structure, specific fusion
  mechanism, specific deployment constraint) is what the new test is designed to let through.
- **Raytheon Company v. Controller General of Patents (Delhi High Court, 2023)** is now embedded
  in the guidelines as policy and is the most-cited precedent in current commentary: it confirms
  that novel hardware is *not* required to overcome the software exclusion, provided the
  technical effect is real. This directly supports filing MBCSS as a software-plus-general-purpose-
  hardware system, which is what your architecture already is.
- **Business methods remain absolutely excluded**, with no technical-effect analysis applied at
  all if the substance is administrative rather than technical — not directly relevant to MBCSS,
  but relevant if the defense-context scoring module (Claim 2) is drafted in a way that reads as
  "a business rule for deciding who to flag" rather than as a technical signal-fusion mechanism.
  This is one more reason to tie Claim 2 tightly to the CNN output (§2.2) rather than to the
  scoring policy itself.

## 2.4 Illustrative direction for narrower claims (not final legal language)

The following reframes your four claims around what the research above suggests is actually
defensible. **This is direction-setting for a conversation with your patent agent, not
ready-to-file language** — claim drafting has legal formalities (antecedent basis, proper
dependency structure, exact statutory phrasing) that need a professional's pass regardless of the
technical content:

- **Claim 1**, narrowed: emphasize the *specific* fusion mechanism (e.g., the learned gating
  fusion from §1.6, if you adopt it) rather than "a fusion layer" generically — a learned,
  confidence-adaptive gate across exactly these three domains is a narrower, more specific
  technical feature than "concatenation followed by dense layers."
- **Claim 2**, narrowed per §2.2: explicitly recite that the risk score's stego-likelihood input
  is the fused output of the Claim 1 network specifically — not "a stego probability" in the
  abstract, which reads as interchangeable with any malware scanner's output score.
- **Claim 3**, narrowed: keep this strictly dependent on Claim 1 ("the system of Claim 1,
  wherein..."), never as an independent claim — standalone INT8-quantization claims have
  essentially no remaining novelty (§2.1).
- **Claim 4**, narrowed: fold this into a dependent method claim describing the *specific*
  pipeline (branches → gated fusion → INT8 deployment → context-fused scoring) applied to
  air-gapped defense networks, rather than an independent claim over "detecting steganographic
  exfiltration," which is prior art as a problem framing (§2.1).

The unifying principle, consistent with what your own Risk Register already suspected: every
claim should read as *inseparable from the specific combination*, because each individual
ingredient — multi-branch fusion, INT8 CPU deployment, weighted risk scoring, and even the
steganalysis-as-insider-threat framing itself — already exists in prior art on its own.

## 2.5 Patent filing logistics — corrections and updates to your budget/timeline

- **Fee correction:** ₹1,750 (as budgeted in `prd.md`) is the **physical/paper filing** fee for
  a natural person/startup/small entity/educational institution, which includes a roughly 10%
  surcharge over e-filing. The **e-filing fee is ₹1,600** for the same applicant categories, and
  e-filing is faster and is what virtually all applicants use — recommend budgeting ₹1,600 and
  filing electronically via the IP India portal rather than on paper.
- **Form 28 is required** to actually claim the concessional (natural person / startup / small
  entity / educational institution) fee rate — file it alongside Form 1 with supporting
  documentation, or the application may default to the large-entity fee schedule (5× higher).
  Check with your college IP cell whether the institution's status affects which category applies
  to a student-led filing.
- **Expedited examination exists and is affordable**: Rule 24C lets startups, individuals, and
  small entities request expedited examination for roughly ₹8,000 (e-filing), cutting the wait
  for a first examination report from the standard 3–5 years down to roughly 1–2 years. Given
  you're filing as students with a graduation timeline, this is worth discussing with your IP
  cell even though it's an added cost beyond the ₹1,600 provisional fee.
- **You'll also need a Digital Signature Certificate or e-Sign** (via Aadhaar/PAN) to e-file —
  budget time to set this up before your "file immediately" pre-Month-0 deadline, since DSC
  issuance isn't instant.

## 2.6 Running your own prior-art search before filing

A patent agent will do a formal search, but you can (and should) do a preliminary pass yourself
before your first conversation with one, so the conversation starts from "here's what I already
found" rather than from zero:

1. **India:** search **InPASS** (Indian Patent Advanced Search System) for existing Indian
   filings in your space.
2. **International:** search **WIPO Patentscope** and **Google Patents** — both are free and
   cover the international patents cited in §2.1.
3. **Relevant classification codes** to search by, rather than only by keyword: **G06T 1/00**
   family (image data processing, including the embedding/hiding-in-images area under this
   subclass), **G06N 3/04x** (neural network architecture, including "combinations of networks"
   under G06N 3/045), and — for the insider-threat/scoring side — **G06F 21/55** (intrusion or
   anomaly detection) and the **H04L 63** range (network security). Searching by these codes
   surfaces filings that don't happen to use your exact keywords.
4. **Search both the architecture and the application separately** — as §2.1 shows, "multi-branch
   fusion CNN" prior art and "steganalysis" prior art live in different literatures and need
   separate searches; the same is true of "risk scoring" (DLP/UEBA literature) versus
   "steganography" (digital forensics literature) for Claim 2.

## 2.7 Updated Risk Register entries

| Risk | What the research adds |
|---|---|
| Patent invalidated by prior art | Confirmed as a real, specific risk — not a generic worry. §2.1–2.2 name the closest prior art directly. The mitigation ("claim the combination, file narrow") in `prd.md` is the right instinct; §2.4 gives it concrete direction. |
| Claim 2 rejected as obvious over existing UEBA/DLP art | New risk to add explicitly: this is now the most exposed of your four claims. Mitigate by tying the risk score's stego-input structurally to Claim 1's specific network output (§2.2), not the scoring formula. |
| CRI Guidelines shift again before filing | Low near-term risk — no evidence of withdrawal or successful challenge as of March 2026 commentary — but guidelines can be revised; confirm current status with your IP cell immediately before filing rather than relying on this document. |

---

# PART 3 — CONSOLIDATED ACTION CHECKLIST

**Do immediately (before/alongside Pre-Month-0):**
- [ ] Budget ₹1,600 (e-filing) not ₹1,750; identify DSC/e-Sign provider; ask IP cell about Form 28 category and expedited examination (§2.5)
- [ ] Reframe the Claim 2 draft around "stego-confidence as a structural input from Claim 1," not the scoring formula itself (§2.2, §2.4)
- [ ] Vectorize `compute_dct_blocks` before Week 2 dataset generation (§1.4)

**Do during Month 1–2 (architecture/branch phase):**
- [ ] Evaluate `jpegio`/`jpeglib` for the J-UNIWARD data path (§1.4)
- [ ] Benchmark a transfer-learned backbone (StegResNet-style) against your from-scratch Branch A (§1.7)
- [ ] Try unfreezing SRM filters during joint fine-tuning as a free ablation row (§1.5)

**Do during Month 2–3 (fusion/quantization phase):**
- [ ] Try the SE-gated fusion as an ablation alternative to fixed concat (§1.6)
- [ ] Measure actual PTQ accuracy drop; add QAT if it exceeds your 5% budget (§1.2)
- [ ] Benchmark ONNX Runtime / OpenVINO against raw PyTorch for the CPU deployment numbers (§1.3)
- [ ] Add resize/recompression augmentation if robustness becomes part of your contribution story (§1.9)

**Do before filing the complete specification (within the 12-month window):**
- [ ] Run a preliminary prior-art search yourself using §2.6 before your IP-cell/patent-agent meeting
- [ ] Walk through the "Seven Stambhas" framing and the two-step algorithm test with whoever drafts final claims (§2.3)
- [ ] Confirm CRI Guidelines haven't been revised since this document (§2.7)

---

## SOURCES

**Efficiency / architecture:**
- Lightweight CNN + HPF for steganalysis (Mar 2026) — researchgate.net/publication/342684818
- PENet+ — arxiv.org/pdf/2606.10939
- Block-wise pruning of EfficientNet-B0 — nature.com/articles/s41598-023-43386-2 ; pmc.ncbi.nlm.nih.gov/articles/PMC10522667
- Green Steganalyzer — arxiv.org/abs/2306.04008
- StegResNet — kuey.net/index.php/kuey/article/download/11054/8608/20464
- CVTStego-Net — sciencedirect.com/science/article/abs/pii/S221421262300279X
- Robustness-to-transformation study (Frontiers in AI) — frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1532895/full
- Gated neural-network fusion — patents.google.com/patent/CN109063765B
- Multi-branch + SE-attention precedent — arxiv.org/pdf/2504.03221 ; pmc.ncbi.nlm.nih.gov/articles/PMC9032940

**Quantization / deployment:**
- PyTorch quantization deprecation notice — docs.pytorch.org/docs/main/quantization.html
- PT2E quantization flow — docs.pytorch.org/ao/stable/pt2e_quantization/
- QAT vs. PTQ accuracy (EfficientNet-B0) — developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt
- OpenVINO / NNCF — medium.com/openvino-toolkit ; edge-ai-vision.com
- Intel Neural Compressor — lightning.ai/docs/pytorch/stable/advanced/post_training_quantization.html
- jpegio — github.com/dwgoon/jpegio
- jpeglib — pypi.org/project/jpeglib

**Patent landscape:**
- Steganography Detection patent family — patents.google.com/patent/US20160042193A1 ; USPTO image-ppubs PDFs for 9,197,655 / 9,589,147 / 9,747,462 / 10,162,976
- Shi et al. steganalysis patent — USPTO image-ppubs PDF 7,496,210
- Texture-image steganalysis patent — USPTO image-ppubs PDF 8,548,262
- Multi-layer fusion CNN (Xerox/Conduent) — patents.google.com/patent/US10068171 ; US20170140253A1
- Multi-task fusion architecture (Samsung) — patents.google.com/patent/US11556784
- User behavior analytics for insider threat detection — USPTO image-ppubs PDFs 11,611,574 / 12,137,110
- Industry DLP/UEBA risk-scoring practice — teramind.co ; fidelissecurity.com ; exabeam.com ; proofpoint.com
- Real-time steganalysis / insider threat framing (Forensic Focus) — forensicfocus.com/articles/real-time-steganalysis

**India patent process / CRI Guidelines 2025:**
- mondaq.com/india/patent/1663246 ; lexology.com (CRI Guidelines summary)
- iam-media.com (IPO CRI Guidelines analysis)
- iiprd.com/navigating-the-new-frontier-indias-revised-cri-guidelines-2025
- iplink-asia.com (Seven Stambhas framework)
- mirandah.com (2025 CRI Guidelines overview)
- intepat.com/blog/cri-guidelines-2025-patent-india-3k (Raytheon precedent, two-step test)
- intepat.com/blog/provisional-patent-application ; intepat.com/blog/patent-fees-cost-india
- incorpx.io/blog/patent-filing-fees-india-2026-startup-vs-company
- legismith.com/patent-fee-calculator-india
- legalclarity.org/how-to-file-a-patent-in-india-forms-fees-and-steps

**Patent classification:**
- uspto.gov CPC scheme G06N / G06T ; cooperativepatentclassification.org definitions G06T / G06V

---

*Document Version: 1.0 | Researched & compiled: September 2026*
*This addendum is general research, not legal or financial advice. Confirm all fee amounts,
guideline status, and claim language with a registered Indian patent agent and your college IP
cell before filing anything.*
