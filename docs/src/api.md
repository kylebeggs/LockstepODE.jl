# API Reference

```@meta
CurrentModule = LockstepODE
```

## Execution Modes

```@docs
LockstepMode
Ensemble
Batched
```

## Memory Ordering

```@docs
MemoryOrdering
PerODE
PerIndex
BatchedOpts
```

## Core Types

```@docs
LockstepFunction
LockstepProblem
LockstepSolution
```

## Integrator Types

```@docs
AbstractLockstepIntegrator
EnsembleLockstepIntegrator
BatchedLockstepIntegrator
```

## CommonSolve Interface

```@docs
CommonSolve.init
CommonSolve.solve
CommonSolve.solve!
SciMLBase.step!
SciMLBase.reinit!
```

## Solution Utilities

```@docs
extract_solutions
extract_at_time
```

## Batched Mode Internals

```@docs
BatchedFunction
SubIntegrator
BatchedSubSolution
_get_idxs
ode_kernel!
create_lockstep_callbacks
batch_u0s
```
