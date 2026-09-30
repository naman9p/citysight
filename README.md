# 🏙️ CitySight — Testing, Validation & Benchmark Performance
**Smart India Hackathon (SIH) — Problem Statement 26127**  
**Project:** CitySight  
**Domain:** Automatic Number Plate Recognition (ANPR), Multi-Camera Vehicle Intelligence & Cross-Camera Tracking

---

## 📖 Overview
**CitySight** is a multi-camera vehicle intelligence system designed for real-world Automatic Number Plate Recognition (ANPR), vehicle tracking, evidence generation, watchlist monitoring, and cross-camera vehicle search.

Unlike a conventional single-frame ANPR pipeline, CitySight combines:
- YOLO-based licence plate detection
- Vehicle detection and plate association
- BYTETrack-based temporal tracking
- Multi-frame quality selection
- Perspective rectification
- PaddleOCR-based text recognition
- Multi-frame OCR fusion
- Indian licence-plate normalization
- Confidence-based decision making (**Accepted / Review / Abstain** output states)
- Vehicle fingerprinting & multi-camera trajectory construction
- Cross-camera candidate search & watchlist alert generation
- Persistent evidence storage & REST-style HTTP API
- Monitoring dashboard

*This document summarizes the testing methodology, experimentally verified performance, expanded validation design, stress testing strategy, latency profiling, scalability targets, and production-readiness criteria used for SIH evaluation.*

---

## 1. Validation Philosophy
CitySight is evaluated at multiple levels rather than relying on a single accuracy number.

| Validation Layer | What is Evaluated |
| :--- | :--- |
| **Unit testing** | Individual functions and modules |
| **Integration testing** | Interaction between detector, OCR, DB, tracker, and APIs |
| **Detector benchmark** | Licence plate localization |
| **OCR benchmark** | Complete licence-plate string recognition |
| **Replay testing** | Real recorded camera feeds |
| **Multi-camera validation** | Same vehicle across different cameras |
| **Edge-case testing** | Blur, glare, rain, angle, darkness, and occlusion |
| **Stress testing** | Multi-stream and high-throughput behaviour |
| **System testing** | Complete end-to-end ANPR workflow |
| **Production evaluation** | Latency, throughput, resource consumption, and reliability |

---

## 2. Evidence Classification
To maintain scientific and engineering integrity, every benchmark value is classified into one of the following categories:

| Label | Meaning |
| :--- | :--- |
| **MEASURED** | Obtained from actual CitySight experiments or test runs |
| **DERIVED** | Mathematically calculated from measured data |
| **BENCHMARK SPEC.** | Proposed larger evaluation dataset/protocol |
| **PROJECTED** | Engineering estimate awaiting hardware benchmark execution |

> **Note:** Projected values must not be presented as experimentally measured values until the corresponding test has actually been executed.

---

## 3. Experimentally Verified Dataset

### 3.1 Detector Validation Dataset
| Metric | Value |
| :--- | :--- |
| Evaluation frames | 445 |
| Ground-truth licence plates | 654 |
| Detector predictions evaluated | 529 |
| IoU thresholds evaluated | 0.50 and 0.30 |
| **Status** | **MEASURED** |

### 3.2 OCR Validation Dataset
| Metric | Value |
| :--- | :--- |
| Ground-truth plate crops | 400 |
| Exact full-string matches | 296 |
| Exact-match failures | 104 |
| **Exact-match accuracy** | **74.00%** |
| **Character Recognition Rate (CRR)** | **94.8%** |
| **Status** | **MEASURED** |

A prediction is considered correct **only** when the entire normalized licence-plate string exactly matches the ground truth.
*   **Correct:** Ground Truth `GJ06LE6897` | Prediction `GJ06LE6897`
*   **Incorrect:** Ground Truth `GJ06LE6897` | Prediction `GJ06LE6891` *(Even a single incorrect character causes the prediction to be marked incorrect).*

#### Character-Level Interpretation
The strict **74.00%** full-string exact-match accuracy and the **94.8%** Character Recognition Rate (CRR) measure two different levels of OCR performance. The observed gap indicates that a significant proportion of the ~26% full-string failures are localized single-glyph errors (e.g., `O ↔ 0`, `B ↔ 8`, `I ↔ 1`, `S ↔ 5`, `Z ↔ 2`) rather than complete OCR breakdowns.

These errors are targeted downstream through:
1. Indian licence-plate regex constraints & plate-format normalization
2. Multi-frame OCR fusion & confidence scoring
3. Temporal agreement across multiple observations (e.g., fixing `GJ06LE6B97` to `GJ06LE6897` based on consensus).

---

## 4. Real-World Camera Validation
Two independently recorded real-world traffic videos were used for end-to-end replay validation.

| Property | CAM_01 | CAM_02 |
| :--- | :--- | :--- |
| **Resolution** | 1080p | 1080p |
| **Frame rate** | 30 FPS | 30 FPS |
| **Recording duration** | ~5 min | ~5 min |
| **Approx. vehicles** | 40–70 | 40–70 |
| **Environment** | Real road traffic | Real road traffic |
| **Replay type** | Offline multi-camera replay | Offline multi-camera replay |

**Example Replay Command:**
```bash
python -m phase2_city.replay \
  --scenario phase2_city/config/replay.step33.yaml \
  --city-config phase2_city/config/city.step33.yaml \
  --config phase1_anpr/config/config.step33.yaml
