# PPA accounting

PPA accounting associates area and leakage with physical hardware, and dynamic energy with the events that use it. Ownership determines the hardware population; the execution schedule determines reuse.

## Hardware ownership

Each component accounts for its own area and leakage once. A composite combines its local costs with its owned components' contributions. A shared source or driver retains one owner even when several components use it.

Physical replication increases hardware cost; temporal reuse increases access events. A child's physical population already includes its enclosing hardware's multiplicity. Grouping operations into one simulation call changes neither count. Virtual measurement channels allocate no hardware, and a channel naming a real child shares that child's accounting entry.

## Access energy

Each contribution belongs to its modeled physical activity. Establishing an array's held boundary is counted once; excursions through successive phases follow the phase schedule. Boundary drivers and the array account for their respective circuit costs.

Numerical iterations used to find an access's operating behavior do not represent additional physical accesses. Chunking partitions the same contributions without multiplying energy.

## Measurement and aggregation

Each retained observation describes one basic operation: one vector for a linear unit or one image for a convolution unit. Modules aggregate the operation's internal work before submission, including physical multiplicity and temporal reuse exactly once. Numerical batching preserves the independent operation positions.

The model owner fixes the observation layout after assembling the hardware and before collection. Different subtrees may retain different granularities; combining their entries elementwise requires an explicit common layout. Applications retain experiment labels and repeat sizes for subsequent analysis.

Static costs are captured after physical setup and counted once across collection contexts. One context covers one model batch and collects dynamic energy and completed operation durations. Measurement preserves electrical behavior and physical sampling without introducing modeled circuit activity.

Collection preserves each name and its observation axes. It retains directly submitted timing and leaves missing data unknown. Reporting derives static energy from the completed observations; applications own grouping, statistical reduction, layout conversion, and presentation.

## Operation duration

Numerical execution and timing analysis describe the same physical schedule. A unit's basic-operation duration includes internal serial work, parallel execution, and pipeline overlap. Batch, token, and timestep extents enumerate independent operations without multiplying that duration.

A unit submits its completed operation duration once per context, over the retained observation positions. Internal phases and numerical chunks neither add unit durations nor alter the external observation layout. Independent logical operator positions own independent hardware units in model evaluation.

Standalone macro experiments submit the macro's modeled complete-operation duration. Paper measurement periods supply separate powered-window overrides during reporting.

## Derived data and aggregation

Working duration describes modeled execution; powered duration describes the supply-on interval used with leakage to calculate static energy. Reporting resolves a component's missing timing from its nearest timed ancestor. Powered windows default to working windows, with explicit overrides resolved independently; an override changes neither recorded work nor raw observations. A child's own duration remains its default powered window unless explicitly replaced.

Names sharing timing must share an observation layout. Reporting retains every sample axis and each hardware name. Missing data remains unknown; zero is a known value. A name's total energy adds its available dynamic and static contributions elementwise while retaining unknown individual fields.

Hardware scope is selected before reporting. Model evaluation includes all hardware allocated to each layer. Independent macro experiments divide local area and leakage by the known macro count, retaining one macro's full complement of readout circuits. Dynamic energy, duration, and the raw collected data stay unchanged. The supplied static costs must describe the hardware represented by one retained energy position; per-scan conversion and paper-component grouping follow in application analysis.

Using working windows as powered windows assumes no powered idle time or power-transition cost. Applications may provide another supply-on schedule; validation converts its paper scan period to one full VMM's powered interval. The task owner composes whole-model latency and cross-unit overlap. Summing unit durations alone does not establish task completion time.
