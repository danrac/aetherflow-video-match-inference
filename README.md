# AetherFlow Video Match Inference Engine

This repository contains the lightweight runtime used by AetherFlow, Premiere Pro, and After Effects.

It owns:

- ONNX Runtime execution
- runtime-compatible feature preparation
- source/reference matching
- timeline reconstruction
- stable match result output
- host-app adapter boundaries

It consumes exported model artifacts described by `contracts/schemas/model_manifest.schema.json` and returns results described by `contracts/schemas/match_result.schema.json`.

## Non-Goals

- dataset acquisition
- synthetic edit generation
- PyTorch training code
- experiment tracking
- checkpoint management

## Initial Module Layout

```text
src/aetherflow_video_match_inference/
  engine.py
  timeline.py
  adapters.py
```

## Runtime Contract

The runtime API should stay small and stable:

```python
match(reference_path, source_paths, model_manifest_path) -> MatchResult
```

For production source-window matching, use the explicit grouped request path:

```python
match_source_windows(SourceWindowMatchRequest(...)) -> MatchResult
```

Host integrations should adapt this result to AetherFlow, Premiere Pro, or After Effects rather than changing the core inference output.

## Storage Root

Generated inference results should live on the FrameFusion volume.

Default root:

```text
/Volumes/FrameFusion/AetherFlow_VideoMatcherData
```

Override when needed:

```bash
export AETHERFLOW_VIDEO_MATCH_DATA_ROOT=/Volumes/FrameFusion/AetherFlow_VideoMatcherData
```

When `--output` is omitted, match results are written to:

```text
$AETHERFLOW_VIDEO_MATCH_DATA_ROOT/inference/match_result_<timestamp>.json
```

## Smoke Command

Run placeholder inference from a model manifest:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli match \
  --model-manifest /Volumes/FrameFusion/AetherFlow_VideoMatcherData/models/aetherflow-video-match-baseline/v0001/model_manifest.json \
  --reference /Volumes/FrameFusion/AetherFlow_VideoMatcherData/datasets/framefusion-smoke-video/v0001/references/sample-video-0001.reference.mp4 \
  --source /Volumes/FrameFusion/AetherFlow_VideoMatcherData/datasets/framefusion-smoke-video/v0001/clips/clip-video-0001.mp4
```

The runtime currently returns deterministic placeholder matches. The shape is the important part: host applications should depend on this match-result contract, not training internals.

Validate that a model manifest points to an ONNX model loadable by ONNX Runtime:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli validate-model \
  --model-manifest /Volumes/FrameFusion/AetherFlow_VideoMatcherData/models/aetherflow-video-match-baseline/v0001/model_manifest.json
```

Run feature-manifest-driven matching:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli match \
  --model-manifest /Volumes/FrameFusion/AetherFlow_VideoMatcherData/models/aetherflow-video-match-baseline/v0001/model_manifest.json \
  --reference /path/to/reference.mp4 \
  --source /path/to/source.mp4 \
  --reference-feature-manifest /path/to/reference.visual-features.json \
  --source-feature-manifest /path/to/source.visual-features.json
```

When feature manifests are provided, the runtime computes a lightweight visual distance from color and optional motion statistics, then emits `feature_manifest_match` reconstruction metadata. `opencv-visual-stats-v5` manifests also contribute luma, edge-density, scene-change, sparse optical-flow, and normalized motion-track statistics. Without feature manifests, it falls back to deterministic placeholder output.

## Source-Window Matching

Validate a grouped source-window request:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli validate-source-window-request \
  --request /path/to/request.json \
  --schema ../contracts/schemas/source_window_match_request.schema.json
```

Run production source-window matching from a request JSON:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli match-source-windows \
  --request /path/to/request.json \
  --output /path/to/match_result.json
```

Build a source-window request from a dataset sample and an inference profile:

```bash
PYTHONPATH=src python3 scripts/build-source-window-match-request.py \
  --profile configs/public-source-generalization-v0002.example.json \
  --dataset-manifest "$AETHERFLOW_VIDEO_MATCH_DATA_ROOT/datasets/public-source-generalization/v0002/dataset_manifest.json" \
  --sample-id reverse-48f-0001-gsfc-20140421-earthorbit-m11525-mobile-fba6c3e2f025 \
  --output /path/to/request.json
```

Run the full source-window smoke chain from a profile:

```bash
PYTHONPATH=src python3 scripts/run-source-window-profile-smoke.py \
  --profile configs/public-source-generalization-v0002.example.json \
  --dataset-manifest "$AETHERFLOW_VIDEO_MATCH_DATA_ROOT/datasets/public-source-generalization/v0002/dataset_manifest.json" \
  --sample-id reverse-48f-0001-gsfc-20140421-earthorbit-m11525-mobile-fba6c3e2f025 \
  --output-dir /path/to/smoke-output \
  --schema ../contracts/schemas/source_window_match_request.schema.json
```

Run a transform-family smoke grid:

```bash
PYTHONPATH=src python3 scripts/run-source-window-profile-smoke-grid.py \
  --profile configs/public-source-generalization-v0002.example.json \
  --dataset-manifest "$AETHERFLOW_VIDEO_MATCH_DATA_ROOT/datasets/public-source-generalization/v0002/dataset_manifest.json" \
  --output-dir /path/to/smoke-grid \
  --schema ../contracts/schemas/source_window_match_request.schema.json \
  --transform reverse \
  --transform simple_cut \
  --transform scale_position \
  --transform picture_in_picture \
  --transform crop \
  --transform letterbox
```

Run cached profile evaluation:

```bash
PYTHONPATH=src python3 scripts/run-profile-cached-eval.py \
  --profile configs/public-source-generalization-v0002.example.json \
  --output /path/to/report.json
```

Run the edge-case fixture inference smoke against the exported baseline:

```bash
PYTHONPATH=src scripts/run-edge-case-smoke.py
```

By default this consumes the v0002 250-sample fixture and writes per-case match results plus a summary to:

```text
/Volumes/FrameFusion/AetherFlow_VideoMatcherData/inference/video-match-edge-case-smoke/v0001/summary.json
```

Create a host-facing payload with timeline edit reconstruction:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli host-payload \
  --match-result /path/to/match_result.json \
  --host aetherflow \
  --output /path/to/host_payload.json
```

The host payload includes the raw match result plus an ordered edit list with frame and second offsets.

Export generic interchange files from a host payload:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli export-edit-json \
  --host-payload /path/to/host_payload.json \
  --output /path/to/edits.json
```

Export an AetherFlow CEP JSON handoff:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli export-cep-json \
  --host-payload /path/to/host_payload.json \
  --output /path/to/aetherflow_cep.json
```

Export a Premiere Pro JSON handoff:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli export-premiere-json \
  --host-payload /path/to/host_payload.json \
  --output /path/to/premiere.json
```

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli export-edl \
  --host-payload /path/to/host_payload.json \
  --output /path/to/timeline.edl
```

Export an After Effects ExtendScript importer:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli export-ae-extendscript \
  --host-payload /path/to/host_payload.json \
  --output /path/to/aetherflow_import.jsx
```
