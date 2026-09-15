# Input slicing

Input slicing decomposes each activation into positional digits. For an unsigned
slice range $[0,R_x-1]$, the activation is reconstructed as

$$X=\sum_{s=0}^{S_x-1}x_sR_x^s.$$

The same numerical decomposition applies to weights and activations. The execution
schedule assigns activation slices to successive macro reads, so $S_x$ increases
read count without increasing programmed macro instance count. An unsliced value
has one digit with positional weight one.

Reconstruction uses the configured shift-adder arithmetic and cost when a hardware
reconstruction circuit is present; otherwise it is a functional weighted sum without
modeled register-width wrap or circuit cost.
