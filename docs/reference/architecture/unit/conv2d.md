# Conv2d mapping

The conv2d operator dimension lowers `F.conv2d` to a matrix multiplication without replicating the programmed kernels. The weight is flattened once; runtime convolution windows form the serial input-vector axis.

## Governing laws

The input is $[B, C_{\mathrm{in}}, H, W]$ and the output is $[B, C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}}]$. A 3-D $[C_{\mathrm{in}}, H, W]$ input is read as $B=1$ and returns a 3-D output, exactly as `F.conv2d`. For kernel $(k_h, k_w)$, stride $(s_h, s_w)$, padding $(p_h, p_w)$, and dilation $(d_h, d_w)$,

$$H_{\mathrm{out}} =
\left\lfloor
\frac{H + 2p_h - d_h(k_h-1) - 1}{s_h}
\right\rfloor + 1,$$

and analogously for $W_{\mathrm{out}}$. The lowered dimensions are

$$K=C_{\mathrm{in}}k_hk_w,\qquad
M=H_{\mathrm{out}}W_{\mathrm{out}},\qquad
N=C_{\mathrm{out}}.$$

**Weight lowering.** Each output-channel kernel is flattened in $(C_{\mathrm{in}}, k_h, k_w)$ row-major order:

$$W_{\mathrm{matrix}}[n,:]
=\operatorname{flatten}(W[n,:,:,:]).$$

The resulting $[N,K]$ matrix contains exactly one copy of every weight.

**Window lowering.** Every output position $(h_o,w_o)$ produces one input vector. Entry $(c_i,i,j)$ reads

$$X\left[c_i,\ h_os_h-p_h+i d_h,\ w_os_w-p_w+j d_w\right],$$

with out-of-bounds positions replaced by zero. Flattening the window in the same $(C_{\mathrm{in}}, k_h, k_w)$ order gives one row of the $[B,M,K]$ input matrix, per batch element.

**Execution and fold.**

$$Y_{\mathrm{matrix}}=X_{\mathrm{windows}}W_{\mathrm{matrix}}^\mathsf{T}$$

produces $[B,M,C_{\mathrm{out}}]$. The window axis is restored to $[H_{\mathrm{out}},W_{\mathrm{out}}]$, the output-channel axis is moved to the front, and the integer bias is added once per output element.

All windows reuse the same programmed weight. Their $M$ axis is a runtime serial-work axis; it is not a weight-replication or physical-instance axis. The substrate remains responsible for input-axis block packing, activation phases, precision slicing, and digital aggregation.

## Noise & non-idealities

Window gather, weight flattening, output folding, and bias addition are exact integer operations. Every deviation enters through the matrix-multiplication substrate, per the [unit family contract](family.md#noise-non-idealities).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $B$ | batch size | — | runtime input axis |
| $C_{\mathrm{in}}, C_{\mathrm{out}}$ | input / output channels | — | `w_logical_shape` |
| $k_h, k_w$ | kernel extent | — | `w_logical_shape` |
| $s_h, s_w$; $p_h, p_w$; $d_h, d_w$ | stride; padding; dilation | — | `stride`, `padding`, `dilation` |
| $K$ | flattened convolution-window length | — | engine logical shape |
| $M$ | output-window count | — | runtime input axis |
| $N$ | output-channel count | — | engine logical shape |
| $b$ | integer bias vector of length $C_{\mathrm{out}}$ | — | `_int_bias` |

## Assumptions, scope & validity

- Operands are integers within the published value ranges; requantization lies outside the unit.
- The input is $[B, C_{\mathrm{in}}, H, W]$, or its 3-D $[C_{\mathrm{in}}, H, W]$ form read as $B=1$; no other rank is accepted, and the output takes the rank of the input.
- Non-zero padding requires the activation value range to contain zero.
- Grouped convolution is out of scope.
