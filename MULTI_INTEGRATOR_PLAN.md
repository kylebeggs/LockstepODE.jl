# Plan: Multi-Integrator Architecture (v2.0 - Keep API)

## Phase 0: Cleanup & Commit Current State
1. Review and clean up callback/MTK work on current branch
2. Commit cleaned state to `mtk` branch
3. Create new branch `multi-integrator` from `mtk`

## Phase 1: Core Architecture Redesign (Keep Types, Change Internals)

**LockstepFunction Changes** (src/core.jl):
- Keep struct name and basic fields
- Add new fields:
  - `sync_policy` (default: `FixedIntervalSync`)
  - `integrators::Vector{ODEIntegrator}` (created during solve)
- Remove fields: `internal_threading` (threading now at integrator level)
- Keep: `wrapper`, `num_odes`, `ode_size`, `callbacks`, `ordering` (may deprecate ordering later)

**Callable Interface Change** (src/core.jl):
- OLD: `(lockstep_func)(du, u, p, t)` - single batched call
- NEW: Not directly callable - instead intercepted by custom `solve()` method
- Integrator coordination happens in solve, not in function call

**Sync Policy Types** (src/core.jl - add new):
```julia
abstract type SyncPolicy end
struct FixedIntervalSync <: SyncPolicy
    interval::Float64
end
```

## Phase 2: Problem Construction (Keep API)

**ODEProblem Constructor** (src/core.jl lines 346-367):
- Keep same signature: `ODEProblem(f::LockstepFunction, u0, tspan, p; kwargs...)`
- Internally: store metadata, don't create single batched problem
- Create N separate `ODEProblem`s (one per ODE) during solve
- Keep: automatic callback wrapping (but simpler internally)

**Batching Utilities** (src/utils.jl):
- Keep: `batch_initial_conditions()` API (for user convenience)
- Internally: split back into vector of u0s for separate problems
- Keep: `extract_solutions()` API
- Internally: merge from N separate solutions

## Phase 3: Solve Method (New Coordinator)

**Custom solve() Dispatch** (src/coordinator.jl - new file):
```julia
function CommonSolve.solve(prob::ODEProblem{<:LockstepFunction}, alg, args...; kwargs...)
    lockstep_func = prob.f

    # Create N separate ODEProblems
    problems = create_individual_problems(lockstep_func, prob.u0, prob.tspan, prob.p)

    # Solve with sync coordination
    solutions = solve_with_sync(problems, alg, lockstep_func.sync_policy; kwargs...)

    # Return merged solution (looks like old batched solution)
    return merge_solutions(solutions, lockstep_func)
end
```

**Sync Coordinator**:
- `solve_with_sync()` - main loop with sync points
- Threading: one task per ODE, barrier at sync intervals
- Adaptive dt per-ODE between syncs
- GPU: each integrator uses GPU arrays independently

## Phase 4: Callback System Simplification

**Remove SubIntegrator** (src/core.jl lines 159-201):
- No longer needed! Each integrator has native callback support
- Per-ODE callbacks just attach to that integrator
- Shared callbacks attach to all integrators

**Simplify create_lockstep_callbacks()** (src/core.jl lines 204-307):
- Much simpler: just distribute callbacks to individual integrators
- No wrapper needed, no cache needed
- Remove 100+ lines of complexity

## Phase 5: MTK Integration (Minimal Changes)

**ext/LockstepODEMTKExt.jl**:
- Keep: `MTKWrapper` unchanged
- Keep: `transform_parameters()` unchanged
- Remove: SubIntegrator SymbolicIndexing code (lines 355-437) - not needed
- Update: ODEProblem constructor to handle new internal structure
- Benefit: `getu()` works natively on each integrator

## Phase 6: GPU Support (Simpler!)

**GPU Extensions** (all ext/*Ext.jl):
- Remove manual kernel launches
- Each integrator receives GPU arrays, runs on GPU automatically
- Sync coordination on CPU (minimal overhead)
- Per-ODE adaptive timestepping **on GPU**

## Phase 7: Testing (Update for New Behavior)

**Test Changes**:
- Keep test file structure
- Update assertions for sync behavior
- Add tests for per-ODE adaptive dt
- Add tests for sync intervals
- GPU tests: verify independent adaptive stepping
- Callback tests: simpler (no SubIntegrator to test)

## Phase 8: Documentation

**README.md**:
- Keep example structure similar
- Add: `sync_policy` parameter documentation
- Highlight: per-ODE adaptive timestepping benefit

**CLAUDE.md**:
- Update architecture section
- Document new multi-integrator internals
- Keep API reference similar

## Key Preserved API Elements
✓ `LockstepFunction` type name and constructor
✓ `ODEProblem(lockstep_func, u0, tspan, p)` constructor
✓ `batch_initial_conditions()` utility
✓ `extract_solutions()` utility
✓ `PerODE` / `PerIndex` ordering (may be less relevant but kept)
✓ Callback API (per-ODE or shared)
✓ MTKWrapper integration

## Key Internal Changes
- Single integrator → N integrators with sync coordination
- Batched callable → custom solve() with coordination
- SubIntegrator wrapper → native integrator callbacks
- Manual GPU kernels → automatic GPU support per integrator
- Forced sync every step → configurable sync intervals

## Benefits
1. Same user-facing API (minimal migration)
2. Per-ODE adaptive timestepping (major performance gain)
3. Per-ODE adaptive on GPU
4. Simpler callback implementation
5. Flexible sync intervals
6. Better for heterogeneous workloads (stiff/non-stiff mix)

## Estimated Impact
- ~80% rewrite of core.jl internals (but keep type names/structure)
- ~60% rewrite of utils.jl (keep function names, change internals)
- New coordinator.jl file (~200 lines)
- Simplify callback system (remove ~100 lines)
- Simplify GPU extensions (remove manual kernels)
- ~70% test updates (same structure, different behavior checks)
- ~50% documentation updates (API similar, benefits different)

## GPU Support Context

OrdinaryDiffEq.jl automatically supports GPU when you pass CUDA vectors:
- Each ODE gets its own `ODEProblem` with GPU arrays (`u0 = cu(...)`)
- Each integrator runs independently on GPU
- Different ODEs can have different adaptive timesteps **on GPU**
- Sync coordination happens on CPU but doesn't require moving data back and forth constantly

Sources:
- [DiffEqGPU.jl Documentation](https://docs.sciml.ai/DiffEqGPU/stable/getting_started/)
- [Massively Data-Parallel GPU Solving](https://docs.sciml.ai/Overview/stable/showcase/massively_parallel_gpu/)

This is actually **superior** to the current batched approach because:
1. **Per-ODE adaptive timestepping on GPU** - stiff ODEs can take smaller steps without slowing down easy ones
2. **Better GPU utilization** - each integrator's GPU kernels can be optimized independently
3. **No forced synchronization** - until sync points, each ODE is completely independent

## Design Decisions (User Preferences)
✓ Fixed time interval sync (simplest, user-controlled via `sync_policy` parameter)
✓ One thread per ODE (best performance, barrier at sync points)
✓ Keep GPU support - works better with multi-integrator!
✓ Breaking v2.0 but keep API surface as similar as possible
