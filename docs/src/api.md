# API Reference

```@meta
CurrentModule = LockstepODE
```

## Types

```@docs
LockstepFunction
PerODE
PerIndex
```

## Functions

```@docs
batch_initial_conditions
extract_solutions
```

## Constructor

```@docs
LockstepFunction(f, ode_size::Int, num_odes::Int; internal_threading=true, ordering=nothing, backend=KA.CPU())
```