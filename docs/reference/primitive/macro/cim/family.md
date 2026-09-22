# CIM macro family

## Shared conventions

A CIM macro stores a logical integer weight matrix and reads one input vector into one code per output at a selected quantization operating point and resolution. Input and weight domains are inclusive continuous integer ranges; redundant encodings leave no holes.

The geometry fixes input capacity $N$, parallel readout-lane count $L$, and serial scan count $S$. Output capacity is

$$M = L S.$$

Logical outputs visit every lane in a scan before advancing to the next scan. Output $j$ occupies lane $j\bmod L$ in scan $\lfloor j/L\rfloor$. This fixed assignment distributes every logical-output prefix as evenly as possible across lanes. Physical cell placement and readout recovery preserve this logical order.

A mapped weight block occupies a valid output prefix of width $n\in[0,M]$, including weight slices, followed by zero-filled padding. Valid zero-valued weights retain their positions and count. Each lane uses a contiguous scan prefix, and lane workloads differ by at most one output.

Disabled word-line selections and closed column branches change the electrical drive conditions. Currents that remain under those conditions retain their conduction cost. Conversion and control events are charged only when executed.

Equal output prefixes and conversion settings give equal access durations. Different prefixes may require different scan counts on the same hardware; their serial or parallel schedule determines the combined duration.

One conversion selects at most $A$ input positions. Unselected positions are zero; a selected position still counts against the limit when its data value is zero. The selection limit bounds the conversion dot-product magnitude used when designing and calibrating the readout.

Each quantization operating point pairs analog references with a calibrated output scale. Conversion depends on those references and the selected resolution.

The macro returns one of two numerical code mappings. Zero-point mapping quantizes into a centered two's-complement code and arithmetically truncates it to the active resolution. Sign-magnitude mapping quantizes the magnitude, truncates it, and restores the sign. Neither mapping is inferred from the logical input or weight ranges.

Finite active resolution satisfies $1\leq b\leq B$, where $B$ is the maximum ADC width. Highest-precision physical execution uses $B$ bits; exact ideal execution omits virtual quantization.

## Governing laws

For a valid logical-output count $n$ and lane index $0\leq\ell<L$, the active scan count is

$$c_\ell(n)=\left\lfloor\frac{n+L-1-\ell}{L}\right\rfloor.$$

The activity of scan $s$ on lane $\ell$ and the complete access duration are

$$m_{\ell,s}(n)=[s<c_\ell(n)],\qquad
t=t_{\mathrm{scan}}\max_\ell c_\ell(n)
=t_{\mathrm{scan}}\left\lceil\frac{n}{L}\right\rceil.$$

An empty prefix has no active scans. Selecting all $M$ outputs uses exactly $S$ scans on every lane. Execution activity and duration follow the same distribution.

A read at finite active ADC resolution $b$ returns one final macro code per output. ADC-internal encoding and sign or zero-point processing remain inside the macro. Its effective rescale factor $s_b$ expresses one output code in MAC units:

$$M_p \approx \mathrm{code}_b\,s_b.$$

Each mode stores one calibrated factor $s_B$ at the maximum ADC resolution $B$. Other active resolutions derive their effective factor through

$$s_b = s_B\,2^{B-b},$$

because dropping one ADC decision bit doubles the full-resolution MAC span represented by one compact code.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $M_p$ | ideal integer dot product of one WL plane | — | — |
| $s_B$ | MAC units per output code at maximum ADC resolution | — | `rescale_factors` |
| $s_b$ | MAC units per output code at active resolution $b$ | — | `rescale_factor` |
| $\mathrm{code}_b$ | final macro code at resolution $b$ | — | `vec_mat_mul` result |
| $b, B$ | active and maximum ADC resolution | — | `adc_active_bits`, `adc_bits` |
| $N$ | configured logical input capacity | — | `input_num` |
| $M$ | derived logical output capacity, $LS$ | — | `output_num` |
| $j,\ell,s$ | logical output, lane, and scan indices | — | — |
| $L$ | parallel readout-circuit groups | — | `lane_num` |
| $S$ | serial output positions per readout group | — | `scan_num` |
| $A$ | maximum selected inputs per conversion | — | `max_active_num` |
| $n$ | valid logical-output count | — | `effective_output_num` |
| $m_{\ell,s}(n)$ | activity of scan $s$ on lane $\ell$ | — | `phase_mask` |
| $c_\ell(n)$ | active scan count on lane $\ell$ | — | — |
| $t_{\mathrm{scan}}$ | duration of one scan | ns | `_latency_per_scan__ns` |
| $t$ | complete operation duration | ns | `latency__ns` |

## Assumptions, scope & validity

TODO (domain author): state the calibrated validity range of the output-rescale relation.
