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
