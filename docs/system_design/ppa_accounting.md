# PPA accounting

A measurement crosses every block that spends energy in one call, and each of them bills alone: an emitter hands a tensor to the ledger and never learns who is measuring, while the measurement never learns which blocks will emit. The layout of that tensor is the whole contract between the two ends, so a maintainer at either end has to know it. This page is the reasoning behind the protocol; the rules an emission site is written to are in [code_style](../conventions/code_style.md).

## Two blocks meeting at a rank

Every energy tensor is laid out `[*caller_leading, ...]`: the measuring caller's own batch or time prefix first, one position per independent unit operation, then whatever structure the emitter's own work has — a serialized round, a digit, a phase, an output slot, a fabrication instance.

The boundary between the two blocks is a rank, and nothing else. That follows from what each end can know. Only the measurement site knows how many leading dims its caller owns, because that is a property of the call being wrapped rather than of any module inside it; only an emitter knows the shape of its own work, which no ledger could enumerate for blocks it has never seen. A rank is the one fact the caller can state once, ahead of the run, without naming a single emitter — so the caller declares it on the profiler, uniformly for the whole measurement, and every emitter is read against that one number. An axis-name registry or a per-emitter declaration would put the same knowledge in two places and make every new emitter an edit to the collector.

The declared rank is a resolution knob over the report, never a total. It sets how finely the per-unit-operation view resolves; a rank of zero is legal and yields scalar records, and the row totals come out identical either way.

## The reduction is deliberately blind

The ledger keeps the leading block and sums every axis past it, distinguishing no kind of axis from another. An emitter therefore declares nothing about its axes and orders them however its math produces them, and giving a block a new internal axis — a second serialized round, one more digit — reaches neither the ledger, nor the reporter, nor any other emitter.

The fold runs where the tensor is built, in the emitter's own frame, so what a record stores is already the per-unit-operation view: one element per unit operation, not a figure collapsed over the batch, and the reporter later sums those elements into rows and a total. Folding at the emitter is also what bounds the book's memory. A flat per-op lump is a 0-dim constant expanded onto the billed layout, stride-0 until the fold collapses it, so an emission allocates only the caller's block; a ledger that stored raw tensors and reduced at report time would instead hold every emitter's full grid for the length of the measurement.

## What a mis-shaped bill does

Neither failure below is a shape error, and neither raises.

- **A pre-reduced caller block.** An emitter that sums its own leading dims hands over a tensor ranked below the declared one. There is then nothing left to keep: the fold finds no axes past the rank, and the record enters the book already collapsed. The row total still comes out right, which is exactly what makes it silent — what is lost is the resolution the protocol exists for, and a book mixing collapsed records with resolved ones can no longer be read position by position.
- **A size-one stand-in for a real extent.** An emitter that opens its tensor with a size-1 axis where a caller extent belongs claims one unit operation where the call performed many. A record is folded alone, never against a second tensor, so there is nothing for that axis to broadcast against and it survives the fold as written: that emitter's row alone comes out short while every other row stays correct, and the total is off by a factor no single row reveals.

This is why the leading block is a true-extent contract rather than a broadcast request. The measurable law on the far side of it is exact proportionality — feeding a model twice the batch doubles every emitter's bill.

## Where in time the bill is taken

A block whose operating point another component settles bills at the settled state, once per access: not while it samples, because a sampling call sees no settled quantity at all, and not inside the iteration that settles it, because one access would then be billed once per step. The delivering call is the seam, and it is also the only place the caller's own layout is available intact.

When the work producing that state runs in chunks, the seam sits after the fold that restores the caller's layout. A chunk axis has ravelled the caller's block into a single axis that can no longer express it, and a padded tail carries duplicate positions on that axis ([compile](compile.md)), so a per-chunk bill both loses the layout and counts padding.

## Billing runs under no grad

Billing is real tensor arithmetic over the same signals the value path computes, so left unguarded it would build an autograd graph of its own. The measured entry points execute under `torch.no_grad()`, and an emitter reachable from a differentiable path builds its tensor under its own guard. The detach applied as a record is laid out is a backstop against a missed guard, not the mechanism — by then the graph it would have to free has already been built.

What is at stake is the side channel's central promise: measuring a run must not change what the run costs. A retained billing graph would keep alive every intermediate the value path had already released, so a measured run and an unmeasured one would differ in peak memory rather than only in bookkeeping.
