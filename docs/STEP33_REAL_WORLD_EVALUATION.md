\# Step 33 — Real-World Two-Camera Replay Evaluation



\## Setup



Two self-recorded 1080p, 30 FPS road videos were replayed through the existing CitySight pipeline.



\- CAM\_01 start: 2026-09-19T12:40:12Z

\- CAM\_02 start: 2026-09-19T12:40:08Z

\- Directed topology: CAM\_01 -> CAM\_02

\- Approximate link distance: 50 m

\- Maximum candidate speed: 60 km/h

\- Replay processing rate: 10 FPS

\- Detector device: CUDA

\- OCR device: CPU

\- Vehicle enrichment: disabled



\## 640 Baseline



\- Observations: 38

\- Accepted: 10

\- Review: 18

\- Abstained: 10

\- CAM\_01 observations: 25

\- CAM\_02 observations: 13



\## Confirmed Real Cross-Camera Match



Source:



\- Camera: CAM\_01

\- Plate: GJ04EJ5933

\- Event ID: acd77414-0947-56d7-86c6-05fe13358cd4

\- Timestamp: 2026-09-19T12:42:06.894521+00:00



Candidate:



\- Camera: CAM\_02

\- Plate: GJ04EJ5933

\- Event ID: 289896b9-07f9-5105-bacf-b52fcb9b0260

\- Timestamp: 2026-09-19T12:42:16.621318+00:00



Candidate decision:



\- Decision: possible\_strong

\- Elapsed time: 9.726797 s

\- Fingerprint similarity: 0.800000

\- Evidence coverage: 0.600000

\- Topology: direct\_link

\- Travel: feasible

\- Minimum travel time at 60 km/h: 3.0 s



For a 50 m link, the implied average speed is approximately 18.5 km/h.



A previous deliberately conflicting source observation produced

`identity\_conflict` while topology and travel remained feasible,

demonstrating negative identity discrimination.



\## Controlled 640 vs 960 Experiment



Only detector image size was changed.



| Metric | 640 | 960 |

|---|---:|---:|

| Observations | 38 | 49 |

| Accepted | 10 | 6 |

| Review | 18 | 24 |

| Abstained | 10 | 19 |

| Raw OCR available | 28 | 30 |

| Normalized plates available | 28 | 28 |



960 produced more observations but did not increase normalized plate

availability and produced more review/abstained observations.



Examples of non-plate OCR at 960 included text such as:



\- PATEL

\- YAHAN POSSIBLE HAI

\- BRAKE GADI PELAGAO



The 960 run also showed weaker CAM\_02 crop/OCR behaviour for some

distant plates.



Therefore the retained Step 33 baseline is:



\- process\_fps: 10

\- image\_size: 640

\- confidence\_threshold: 0.35

\- iou\_threshold: 0.45



\## Conclusion



The real-world Step 33 experiment demonstrates a successful genuine

CAM\_01 -> CAM\_02 association for the same vehicle using identity,

topology, and travel-feasibility evidence.



The 960 image-size experiment did not improve the current end-to-end

ANPR result, so 640 remains the baseline.

