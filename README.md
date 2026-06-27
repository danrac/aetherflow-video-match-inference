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

Run feature-manifest-driven matching:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli match \
  --model-manifest /Volumes/FrameFusion/AetherFlow_VideoMatcherData/models/aetherflow-video-match-baseline/v0001/model_manifest.json \
  --reference /path/to/reference.mp4 \
  --source /path/to/source.mp4 \
  --reference-feature-manifest /path/to/reference.visual-features.json \
  --source-feature-manifest /path/to/source.visual-features.json
```

When feature manifests are provided, the runtime computes a lightweight visual distance from color and optional motion statistics, then emits `feature_manifest_match` reconstruction metadata. `opencv-visual-stats-v3` manifests also contribute luma, edge-density, scene-change, and sparse optical-flow statistics. Without feature manifests, it falls back to deterministic placeholder output.

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
