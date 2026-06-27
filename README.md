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

## Smoke Command

Run placeholder inference from a model manifest:

```bash
PYTHONPATH=src python3 -m aetherflow_video_match_inference.cli match \
  --model-manifest /private/tmp/aetherflow-video-match-smoke/model/model_manifest.json \
  --reference /private/tmp/aetherflow-video-match-smoke/dataset/references/sample-0001.reference.txt \
  --source /private/tmp/aetherflow-video-match-smoke/dataset/clips/clip-0001.txt \
  --output /private/tmp/aetherflow-video-match-smoke/match_result.json
```

The runtime currently returns deterministic placeholder matches. The shape is the important part: host applications should depend on this match-result contract, not training internals.
