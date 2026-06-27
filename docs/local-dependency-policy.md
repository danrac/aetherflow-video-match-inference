# Local Dependency Policy

Before adding ONNX Runtime, OpenCV, PyAV, FFmpeg, or host-runtime dependencies, check the shared local inventory at:

```text
/Users/danracusin/Documents/AetherFlow_VideoMatcherModel/docs/local-dependency-inventory.md
```

The AetherFlow CEP runtime already includes `onnxruntime`, `torch`, `cv2`, `av`, `numpy`, and `pydantic`, plus a bundled FFmpeg binary. Prefer using that runtime for inference compatibility work unless a narrower runtime is intentionally required.

Use `/Volumes/FrameFusion/AetherFlow_VideoMatcherData` for match results, reconstruction artifacts, caches, and other generated inference outputs. Override with `AETHERFLOW_VIDEO_MATCH_DATA_ROOT` when needed.

Do not add heavyweight inference dependencies until the inventory has been refreshed and the missing dependency is documented.
