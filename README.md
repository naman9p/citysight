# 🏙️ CitySight — Testing, Validation & Benchmark Performance
**Smart India Hackathon (SIH) — Problem Statement 26127**  
**Project:** CitySight  
**Domain:** Automatic Number Plate Recognition (ANPR), Multi-Camera Vehicle Intelligence & Cross-Camera Tracking

## Project Documentation & Validation

**Source Repository:** [github.com/naman9p/citysight](https://github.com/naman9p/citysight)

| Validation Metric | Result |
| :--- | :---: |
| **Detector Precision** | **92.68%** @ IoU 0.50 |
| **Detector Recall** | **94.8%** @ IoU 0.50 |
| **OCR Exact Match** | **91.53%** · 649 GT crops |
| **Automated Tests** | **691 Passed** |

## Core Technical Research & References

- **Multi-Object Tracking (BYTETrack):** Y. Zhang and research team (2022)
- **Deep Learning OCR (PaddleOCR 3.7):** Y. Du and team (2020)
- **Edge Detection Methodology (YOLO):** G. Jocher and the Ultralytics team
- **Vision Processing (OpenCV):** G. Bradski (2000)

## Key Takeaways

- Works with **Indian / BH plate format**
- **Edge-Optimised ANPR pipeline**
- **Real-world validated**
- **Multi-camera ANPR & vehicle tracking**

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
- GIS-aware camera topology and trajectory visualization
- Macro traffic analytics with camera-density / heatmap-ready output
- Origin-destination (OD) aggregation from accepted trajectories
- Rule-based route-anomaly alerts using topology and travel-time feasibility
- Persistent evidence storage & REST-style HTTP API
- Monitoring dashboard

*This document summarizes the testing methodology, experimentally verified performance, expanded validation design, stress testing strategy, latency profiling, scalability targets, and production-readiness criteria used for SIH evaluation.*

### SIH 26127 Capability Coverage

| Requirement | CitySight implementation status |
| :--- | :--- |
| **>90% recognition accuracy** | **Detector Recall: 94.8% @ IoU 0.50** · **OCR Exact Match: 91.53%**. |
| **Single-plate multi-camera trajectory** | **Implemented** — accepted sightings are reconstructed chronologically with timestamps and camera transitions. |
| **GIS movement history** | **Implemented** — camera latitude/longitude, road, heading, directed topology and a dashboard camera map are available. |
| **Macro traffic analytics** | **Implemented (MVP)** — camera-wise observed traffic volume, accepted counts, hourly flow and normalized heatmap weights. |
| **Origin–destination patterns** | **Implemented (MVP)** — OD camera-pair aggregation and transition counts from accepted multi-camera trajectories. |
| **Route-anomaly alerts** | **Implemented (MVP)** — explainable topology-gap and implausible-travel-speed alerts. |
| **Watchlist / blacklist alerts** | **Implemented** — watchlist matching, de-duplication, persistence and dashboard/API exposure. |
| **Cross-camera candidate evaluation** | **Implemented evaluator** — Recall@K, HitRate@K and MRR are supported when explicit external ground truth is supplied. |

#### City Analytics API

```text
GET /v1/analytics/traffic
GET /v1/analytics/origin-destination
GET /v1/alerts/route-anomalies
GET /v1/cameras
GET /v1/links
POST /v1/trajectories
```

The analytics endpoints are deterministic and read-only. They operate on persisted CitySight observations and the configured city-camera topology; they do not invent vehicle identities or ground-truth labels.

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

## 3. Validation Dataset

### 3.1 Detector Validation Dataset
| Metric | Value |
| :--- | :--- |
| Evaluation frames | 445 |
| Ground-truth licence plates | **654** |
| Detector predictions evaluated | **669** |
| True Positives | **620** |
| False Positives | **49** |
| False Negatives | **34** |
| **Detector Precision @ IoU 0.50** | **92.68%** |
| **Detector Recall @ IoU 0.50** | **94.8%** |

**Consistency check:** Precision = 620 / 669 = **92.68%** · Recall = 620 / 654 = **94.80%**.

### 3.2 OCR Validation Dataset
| Metric | Value |
| :--- | :--- |
| Ground-truth plate crops | **649** |
| Exact full-string matches | **594** |
| Exact-match failures | **55** |
| **OCR Exact Match** | **91.53%** |

**Consistency check:** 594 / 649 = **91.5254%**, reported as **91.53%**.

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
