# FastDepthNet encoder-decoder performance analysis (Bolt OpenCL vs ImmersiveWorld Vulkan)

This guide compares **Bolt's GPU inference path** (OpenCL/GCL backend) with a custom **Vulkan backend** for FastDepth-style encoder-decoder models.

> Note: this is a code-path and methodology analysis from Bolt sources, not measured numbers from your Vulkan runtime.

## 1) What Bolt is doing that affects FastDepth-like models

FastDepth-like networks are dominated by depthwise/pointwise convolutions and decoder upsampling blocks. In Bolt:

1. **Depthwise, pointwise, and depthwise+pointwise are explicitly supported in GPU convolution path**.
   - `ConvolutionOCL` dispatches separate paths for `CONVOLUTION_DEPTHWISE`, `CONVOLUTION_POINTWISE`, and `CONVOLUTION_DEPTHWISE_POINTWISE`.
2. **Kernel algorithm auto-tuning and cache map exist**.
   - `infer_forward_algorithm` uses `CONVOLUTION_TUNNING`, and stores/loads per-layer kernel choices through `AlgorithmMap`.
3. **Deconvolution limitations matter for decoder design**.
   - Bolt logs that GPU does **not support depthwise deconvolution** (`group > 1`); it recommends replacing with standard deconvolution or resize.
4. **Benchmark path includes warmup, explicit GPU finish, and optional algorithm map save/load**.
   - This directly impacts first-run vs steady-state latency comparability.
5. **C API serializes some GPU sections with a global mutex**.
   - This can reduce multi-model or multi-thread throughput if you compare against a Vulkan runtime with finer-grained synchronization.

## 2) Practical comparison matrix for your Vulkan backend

For each row, capture both latency and GPU timeline evidence.

| Topic | Bolt behavior | Vulkan backend check |
|---|---|---|
| Depthwise/pointwise fusion | Has dedicated depthwise+pointwise path | Verify you fuse equivalent blocks (or use subgroup-friendly kernels) |
| Algorithm tuning | Runtime tuning + persisted `algorithmMapPath` | Ensure pipeline/key tuning cache persists across runs |
| Decoder upsampling | Depthwise deconv unsupported on GPU | Prefer resize+conv or non-depthwise transposed conv in both stacks for fairness |
| Warmup/steady split | Warmup loop + timed loop | Use the same warmup count and timing boundaries |
| Synchronization scope | Global GPU mutex in C API code paths | Check if your Vulkan queues are over-synchronized (or Bolt is serialized more) |
| Output readback | GPU benchmark path may map/copy output | Make sure both paths include or exclude readback consistently |

## 3) Recommended apples-to-apples benchmark protocol

1. Use the same model topology, precision, and input tensor shape.
2. In Bolt, run with GPU affinity and algorithm map persistence:

```bash
./benchmark -m fastdepth.bolt -a GPU -w 20 -l 200 -p fastdepth_algo.map
```

3. Run a second time reusing the saved algorithm map (to remove first-tune noise).
4. In Vulkan, run identical warmup/loop counts and include a fully synchronized end-of-iteration marker.
5. Record:
   - End-to-end median / P90 latency
   - Per-layer timing (or per-dispatch timing)
   - GPU occupancy / wavefront utilization
   - DRAM read/write bandwidth

## 4) Where differences usually come from in FastDepth-style models

1. **Depthwise convolution kernel quality**
   - If Vulkan depthwise shaders are scalarized or poorly vectorized, they lose quickly versus tuned OpenCL kernels.
2. **Memory layout and conversion overhead**
   - Any extra image/buffer conversion between encoder and decoder stages can dominate latency.
3. **Decoder strategy mismatch**
   - If one side uses resize+conv and the other uses transposed conv, results are not directly comparable.
4. **Synchronization and queue submission overhead**
   - Frequent queue waits/fences can erase gains from faster kernels.
5. **Autotuning lifecycle**
   - Comparing Bolt's tuned steady-state against Vulkan's untuned first-run is misleading.

## 5) FastDepth-specific fairness checklist

- Same input resolution (for example 224x224 or 320x240).
- Same precision mode (FP16 vs FP32; avoid mixed hidden differences).
- Same post-processing path included/excluded from timing.
- Same batch size (usually 1 for real-time depth).
- Same thermal state and power mode during runs.

## 6) Interpreting outcomes quickly

- If Bolt wins mostly in encoder blocks: inspect depthwise and pointwise kernel occupancy.
- If Bolt wins mostly in decoder blocks: inspect upsampling method and memory traffic.
- If per-layer times are close but E2E differs: inspect synchronization and readback boundaries.

---

If you share one benchmark dump from Bolt and one Vulkan GPU timeline, this file can be extended into a precise bottleneck-by-bottleneck diagnosis.
