# Local Dependency Policy

Before adding ONNX Runtime, OpenCV, PyAV, FFmpeg, or host-runtime dependencies, check the shared local inventory at:

```text
/Users/danracusin/Documents/AetherFlow_VideoMatcherModel/docs/local-dependency-inventory.md
```

The AetherFlow CEP runtime already includes `onnxruntime`, `torch`, `cv2`, `av`, `numpy`, and `pydantic`, plus a bundled FFmpeg binary. Prefer using that runtime for inference compatibility work unless a narrower runtime is intentionally required.

Do not add heavyweight inference dependencies until the inventory has been refreshed and the missing dependency is documented.
