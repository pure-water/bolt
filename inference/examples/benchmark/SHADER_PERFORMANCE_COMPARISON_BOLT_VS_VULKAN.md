# Shader performance comparison: Bolt-style vs Vulkan custom

This note compares two compute shader variants for **`dw_3x3_s2`** (depthwise 3x3, stride-2) workloads on Mali-G78.

## 1) Source code

### 1.1 Your Vulkan compute shader (`dw_3x3_s2`, as provided)

Operation intent for this shader:
- **Type**: depthwise convolution
- **Kernel**: 3x3
- **Stride**: 2 (via `inX=x*2`, `inY=y*2`)
- **Channel mapping**: depthwise (`input channel cN` multiplied by `weight channel cN`)

```glsl
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set=0,binding=0) uniform sampler2DArray uInput;
layout(set=0,binding=1) uniform isampler2DArray uWeight;
layout(set=0,binding=2) readonly buffer BiasBuf{float bias[];};
layout(set=0,binding=3) readonly buffer ScaleBuf{float scaleW[];};
layout(set=0,binding=4) readonly buffer BnMulBuf{float bnMul[];};
layout(set=0,binding=5) readonly buffer BnAddBuf{float bnAdd[];};
layout(set=0,binding=6,rgba32f) uniform writeonly image2DArray uOutput;
layout(push_constant) uniform Push { int width; int height; int inChannels; int outChannels; int fuseRelu; } pc;
void main(){
 int x=int(gl_GlobalInvocationID.x), y=int(gl_GlobalInvocationID.y), g=int(gl_GlobalInvocationID.z); if(x>=pc.width||y>=pc.height) return;
 int c0=g*4+0,c1=g*4+1,c2=g*4+2,c3=g*4+3; if(c0>=pc.outChannels) return;
 ivec2 inSize=textureSize(uInput,0).xy; int inX=x*2,inY=y*2;
 float a0=bias[c0],a1=(c1<pc.outChannels)?bias[c1]:0.0,a2=(c2<pc.outChannels)?bias[c2]:0.0,a3=(c3<pc.outChannels)?bias[c3]:0.0;
 for(int ky=0;ky<3;ky++) for(int kx=0;kx<3;kx++){
  int ix=inX+kx-1, iy=inY+ky-1; if(ix<0||iy<0||ix>=inSize.x||iy>=inSize.y) continue;
  a0 += texelFetch(uInput,ivec3(ix,iy,c0),0).r * float(texelFetch(uWeight,ivec3(kx,ky,c0),0).r);
  if(c1<pc.outChannels) a1 += texelFetch(uInput,ivec3(ix,iy,c1),0).r * float(texelFetch(uWeight,ivec3(kx,ky,c1),0).r);
  if(c2<pc.outChannels) a2 += texelFetch(uInput,ivec3(ix,iy,c2),0).r * float(texelFetch(uWeight,ivec3(kx,ky,c2),0).r);
  if(c3<pc.outChannels) a3 += texelFetch(uInput,ivec3(ix,iy,c3),0).r * float(texelFetch(uWeight,ivec3(kx,ky,c3),0).r);
 }
 float y0=a0*scaleW[c0],y1=(c1<pc.outChannels)?a1*scaleW[c1]:0.0,y2=(c2<pc.outChannels)?a2*scaleW[c2]:0.0,y3=(c3<pc.outChannels)?a3*scaleW[c3]:0.0;
 y0=y0*bnMul[c0]+bnAdd[c0]; if(c1<pc.outChannels)y1=y1*bnMul[c1]+bnAdd[c1]; if(c2<pc.outChannels)y2=y2*bnMul[c2]+bnAdd[c2]; if(c3<pc.outChannels)y3=y3*bnMul[c3]+bnAdd[c3];
 if(pc.fuseRelu!=0){y0=clamp(y0,0.0,6.0);y1=clamp(y1,0.0,6.0);y2=clamp(y2,0.0,6.0);y3=clamp(y3,0.0,6.0);} 
 imageStore(uOutput, ivec3(x,y,g), vec4(y0,y1,y2,y3));
}
```

### 1.2 Bolt-transferred Vulkan compute shader (comparison `dw_3x3_s2` shader)

Operation intent for this shader:
- **Type**: depthwise convolution
- **Kernel**: 3x3
- **Stride**: 2 (effective vertical stride from `(idy << 1)` and horizontal stride via `idx * pc.sw` where `pc.sw=2` in this comparison setup)
- **Channel mapping**: depthwise (`input channel cN` multiplied by `weight channel cN`)

```glsl
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set=0,binding=0) uniform sampler2DArray uInput;
layout(set=0,binding=1) uniform sampler2DArray uWeightF32;
layout(set=0,binding=2) readonly buffer BiasBuf { float bias4[]; };
layout(set=0,binding=3, rgba32f) uniform writeonly image2DArray uOutput;
layout(push_constant) uniform Push {
    int ow;
    int oh;
    int oc;
    int sw;
    int iw_off;
    int ih_off;
    int fuseRelu;
} pc;

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int idy = int(gl_GlobalInvocationID.y);
    int gid = int(gl_GlobalInvocationID.z);

    if (idx >= pc.ow || idy >= pc.oh) return;

    int oc4 = (pc.oc + 3) >> 2;
    int idz = gid % oc4;
    int c0 = idz * 4 + 0;
    int c1 = idz * 4 + 1;
    int c2 = idz * 4 + 2;
    int c3 = idz * 4 + 3;
    if (c0 >= pc.oc) return;

    int in_off_x = idx * pc.sw + pc.iw_off;
    int in_off_y = (idy << 1) + pc.ih_off;

    vec4 acc = vec4(
        bias4[c0],
        (c1 < pc.oc) ? bias4[c1] : 0.0,
        (c2 < pc.oc) ? bias4[c2] : 0.0,
        (c3 < pc.oc) ? bias4[c3] : 0.0
    );

    for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
            int ix = in_off_x + kx;
            int iy = in_off_y + ky;

            float v0 = texelFetch(uInput, ivec3(ix, iy, c0), 0).r;
            float w0 = texelFetch(uWeightF32, ivec3(kx, ky, c0), 0).r;
            acc.x += v0 * w0;

            if (c1 < pc.oc) {
                float v1 = texelFetch(uInput, ivec3(ix, iy, c1), 0).r;
                float w1 = texelFetch(uWeightF32, ivec3(kx, ky, c1), 0).r;
                acc.y += v1 * w1;
            }
            if (c2 < pc.oc) {
                float v2 = texelFetch(uInput, ivec3(ix, iy, c2), 0).r;
                float w2 = texelFetch(uWeightF32, ivec3(kx, ky, c2), 0).r;
                acc.z += v2 * w2;
            }
            if (c3 < pc.oc) {
                float v3 = texelFetch(uInput, ivec3(ix, iy, c3), 0).r;
                float w3 = texelFetch(uWeightF32, ivec3(kx, ky, c3), 0).r;
                acc.w += v3 * w3;
            }
        }
    }

    if (pc.fuseRelu != 0) {
        acc = clamp(acc, 0.0, 6.0);
    }

    imageStore(uOutput, ivec3(idx, idy, idz), acc);
}
```

## 2) Compiler reports (as provided)

### 2.1 Your shader report

```text
Mali Offline Compiler v7.6.0 (Build 72a3cc)
Hardware: Mali-G78 r1p1
Shader type: Vulkan Compute

Work registers: 26 (100% occupancy)
Uniform registers: 20
Shared storage: 0 bytes
Stack spilling: false
16-bit arithmetic: 0%

                              FMA     CVT     SFU      LS       T    Bound
Total instruction cycles:    0.17    1.83    0.56    6.00    4.50       LS
Shortest path cycles:        0.00    0.09    0.00    0.00    0.00      CVT
Longest path cycles:         0.17    1.83    0.56    6.00    4.50       LS

Shader properties
Has uniform computation: true
```

### 2.2 Bolt-transferred shader report

```text
Mali Offline Compiler v7.6.0 (Build 72a3cc)
Hardware: Mali-G78 r1p1
Shader type: Vulkan Compute

Work registers: 58 (50% occupancy)
Uniform registers: 18
Shared storage: 0 bytes
Stack spilling: false
16-bit arithmetic: 0%

                              FMA     CVT     SFU      LS       T    Bound
Total instruction cycles:    0.64    4.06    0.44    6.00   18.00        T
Shortest path cycles:        0.00    0.09    0.00    0.00    0.00      CVT
Longest path cycles:         0.62    4.05    0.44    6.00   18.00        T

Shader properties
Has uniform computation: true
```

## 3) Conclusion

1. **For this Vulkan/G78 compile comparison, your shader compiles better** than the Bolt-transferred shader:
   - lower work registers (26 vs 58),
   - higher occupancy (100% vs 50%),
   - significantly lower texture-cycle pressure (T=4.5 vs T=18.0).

2. **The Bolt-transferred Vulkan shader appears more texture-bound**, which is consistent with slower behavior for large decoder stages.

3. **In this environment, your kernel is likely the better baseline**. Suggested next step is incremental optimization on your kernel (packed vec4 fetch path + layer-specific workgroup tuning), rather than trying to directly transplant Bolt OpenCL kernels.

4. This result does **not** imply native Bolt OpenCL is always slower; it means this specific *OpenCL-to-Vulkan transferred* comparison shader compiles worse than your current Vulkan implementation on the tested compiler target.
