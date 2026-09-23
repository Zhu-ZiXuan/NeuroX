# Conv2d mapping

Convolution lowers to one independent matrix multiplication per channel group, with one copy of each kernel. Flattened convolution windows supply successive input vectors.

## Governing laws

The input is $[\ldots, C_{\mathrm{in}}, H, W]$ and the output is $[\ldots, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}}]$. The leading axes enumerate independent images and are preserved in order. A 3-D $[C_{\mathrm{in}}, H, W]$ input contains one image with no leading axes and returns a 3-D output. Every image reuses the same programmed kernels. For kernel $(k_h, k_w)$, stride $(s_h, s_w)$, padding $(p_h, p_w)$, and dilation $(d_h, d_w)$,

$$H_{\mathrm{out}} =
\left\lfloor
\frac{H + 2p_h - d_h(k_h-1) - 1}{s_h}
\right\rfloor + 1,$$

and analogously for $W_{\mathrm{out}}$. With $G$ groups, both channel counts are divisible by $G$. The weight shape is $[C_{\mathrm{out}},C_{\mathrm{in}}/G,k_h,k_w]$, and the lowered dimensions per group are

$$K=\frac{C_{\mathrm{in}}}{G}k_hk_w,\qquad
M=H_{\mathrm{out}}W_{\mathrm{out}},\qquad
N=\frac{C_{\mathrm{out}}}{G}.$$

**Weight lowering.** Each output-channel kernel is flattened in $(C_{\mathrm{in}}/G, k_h, k_w)$ row-major order within its group:

$$W_{\mathrm{matrix}}[g,n,:]
=\operatorname{flatten}(W[gN+n,:,:,:]).$$

The resulting $[G,N,K]$ tensor contains exactly one copy of every weight.

**Window lowering.** Every output position $(h_o,w_o)$ produces one input vector per group. Entry $(c_i,i,j)$ in group $g$ reads

$$X\left[g\frac{C_{\mathrm{in}}}{G}+c_i,\ h_os_h-p_h+i d_h,\ w_os_w-p_w+j d_w\right],$$

with out-of-bounds positions replaced by zero. Flattening each group's window in the same order gives a $[\ldots,M,G,K]$ input tensor, retaining the image-leading axes.

**Execution and fold.**

$$Y_{\mathrm{matrix}}[\ldots,:,g,:]=X_{\mathrm{windows}}[\ldots,:,g,:]W_{\mathrm{matrix}}[g,:,:]^\mathsf{T}$$

produces $[\ldots,M,G,N]$. Group outputs are concatenated into $C_{\mathrm{out}}$ channels without summation. The window axis is restored to $[H_{\mathrm{out}},W_{\mathrm{out}}]$, the output-channel axis is placed immediately before those spatial axes, and the integer bias is added once per output element.

The $M$ windows reuse the programmed weight through the [CIM input pipeline](cim.md#input-pipeline). Local work overlaps its predecessor's global processing, with startup and drain counted once per image. The number and extents of image-leading axes change neither hardware population nor one image's duration.

Groups own independent macros and local and global recovery circuits and execute in parallel. Area, leakage and dynamic energy sum across groups; image latency is the maximum group latency. Equal group geometry and a common configuration give identical schedules, so group count does not multiply the per-group duration. Unit-local peripheral costs retain their whole-unit meaning.

Input-slot merge operates within each group. For depth-wise convolution, $G=C_{\mathrm{in}}$, $K=k_hk_w$, and $N$ is the channel multiplier. Merge reduces macro count only when a group has multiple output tiles and a macro can hold multiple input slots. Different depth-wise groups never share a macro. A point-wise kernel uses the same mapping with $k_h=k_w=1$.

## Noise & non-idealities

Window gather, weight flattening, output folding, and bias addition are exact integer operations. Every deviation enters through the matrix-multiplication substrate, per the [unit family contract](family.md#noise-non-idealities).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $C_{\mathrm{in}}, C_{\mathrm{out}}$ | total input / output channels | — | `w_logical_shape`, `groups` |
| $G$ | independently mapped convolution groups | — | `groups` |
| $k_h, k_w$ | kernel extent | — | `w_logical_shape` |
| $s_h, s_w$; $p_h, p_w$; $d_h, d_w$ | stride; padding; dilation | — | `stride`, `padding`, `dilation` |
| $K$ | flattened convolution-window length per group | — | flattened kernel shape |
| $M$ | output-window count | — | runtime input axis |
| $N$ | output-channel count per group | — | flattened kernel shape |
| $b$ | integer bias vector of length $C_{\mathrm{out}}$ | — | `_int_bias` |

## Assumptions, scope & validity

- Non-zero padding requires the activation value range to contain zero.
- Group count is positive and divides both channel counts.
