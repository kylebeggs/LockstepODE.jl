# Heterogeneous ODE Functions Design Document

## Objective

Enable LockstepODE.jl to solve different ODE functions (f₁, f₂, f₃, ...) in lockstep, rather than just multiple instances of the same ODE function with different parameters/initial conditions.

**Current Capability**: Solve N instances of the same ODE `f(du, u, p, t)` in parallel
```julia
# Same function, different parameters
lockstep_func = LockstepFunction(lorenz!, 3, 100)  # 100 Lorenz systems
```

**Desired Capability**: Solve N different ODEs in parallel, synchronized in time
```julia
# Different functions, potentially different sizes
fs = [exponential_decay!, harmonic_oscillator!, lorenz!]
sizes = [1, 2, 3]
lockstep_func = LockstepFunction(fs, sizes)  # 1+2+3 = 6 total state variables
```

## Key Benefits

1. **Multi-physics simulations**: Couple different physical systems (mechanical + electrical + thermal)
2. **Ensemble methods**: Test different models against same forcing
3. **Hybrid systems**: Mix discrete and continuous dynamics
4. **Model comparison**: Run competing models in parallel
5. **Memory efficiency**: Share time integration overhead across heterogeneous systems

## Architectural Analysis

### Current Architecture Assumptions

The existing implementation makes several assumptions about uniformity:

1. **Single Function**: `f::F` stored once, called for all ODEs
2. **Fixed Size**: `ode_size::Int` applies to all ODEs
3. **Uniform Indexing**: `_get_idxs` uses arithmetic based on fixed size
4. **Batching**: Utilities assume all ODEs have same dimensionality

### Design Constraints

1. **Time Synchronization**: Must preserve lockstep property (same `t` for all ODEs)
2. **Parallel Execution**: Must work with both CPU threading and GPU kernels
3. **Callback System**: Per-ODE callbacks must still work correctly
4. **Performance**: Uniform case should not regress
5. **Memory Layouts**: Support both PerODE and PerIndex ordering

## Challenges and Solutions

### Challenge 1: Variable ODE Sizes

**Problem**: Different ODEs may have different state dimensions
- f₁: 1 variable (exponential decay)
- f₂: 2 variables (harmonic oscillator)
- f₃: 3 variables (Lorenz system)

Current indexing scheme: `((i - 1) * ode_size + 1):(i * ode_size)` assumes fixed size.

**Solution**:
- Store per-ODE sizes: `ode_sizes::Vector{Int}`
- Compute cumulative offsets: `offsets = cumsum([0; ode_sizes[1:end-1]])`
- Index using: `(offsets[i] + 1):(offsets[i] + ode_sizes[i])`

**PerODE Layout** (straightforward):
```
ODE 1 (size 1): [u₁]
ODE 2 (size 2): [u₁, u₂]
ODE 3 (size 3): [u₁, u₂, u₃]
Memory: [ODE1_u₁, ODE2_u₁, ODE2_u₂, ODE3_u₁, ODE3_u₂, ODE3_u₃]
```

**PerIndex Layout** (complex):
```
Index 1 across all: [ODE1_u₁, ODE2_u₁, ODE3_u₁]
Index 2 across all: [ODE2_u₂, ODE3_u₂]  (skip ODE1, size=1)
Index 3 across all: [ODE3_u₃]            (skip ODE1,2)
Memory: [ODE1_u₁, ODE2_u₁, ODE3_u₁, ODE2_u₂, ODE3_u₂, ODE3_u₃]
```

Requires pre-computed mapping structure to handle ragged indices.

### Challenge 2: Function Storage and Dispatch

**Problem**: Need to store and call different functions for each ODE.

**Options Considered**:

**Option A - Vector{Function}** (SELECTED):
```julia
struct LockstepFunction{O, C}
    fs::Vector{Function}
    ode_sizes::Vector{Int}
    num_odes::Int
    # ...
end
```
**Pros**:
- Simple implementation
- Flexible (runtime determined)
- Easy to add/remove ODEs

**Cons**:
- Type-unstable (dynamic dispatch)
- Small performance overhead vs uniform case

**Option B - Tuple of Functions**:
```julia
struct LockstepFunction{O, F<:Tuple, C}
    fs::F  # NTuple{N, Function}
    ode_sizes::NTuple{N, Int}
    # ...
end
```
**Pros**:
- Type-stable
- Generated function dispatch possible
- Maximum performance

**Cons**:
- Complex implementation
- Compile-time number of ODEs
- Less flexible

**Decision**: Use Vector{Function} approach. The dynamic dispatch overhead is acceptable given:
- Time integration dominates cost (not function call)
- Flexibility valuable for users
- Simpler implementation and testing
- Can optimize later if needed (FunctionWrappers.jl)

### Challenge 3: Indexing System Rewrite

**Problem**: Current `_get_idxs` assumes uniform size arithmetic.

**Solution**:

**For PerODE**:
```julia
function _get_idxs(lockstep_func::LockstepFunction{PerODE}, i)
    offset = sum(lockstep_func.ode_sizes[1:i-1]; init=0)
    return (offset + 1):(offset + lockstep_func.ode_sizes[i])
end
```

**For PerIndex** (complex):
```julia
# Pre-compute at construction time
function _compute_perindex_mapping(ode_sizes::Vector{Int}, num_odes::Int)
    max_size = maximum(ode_sizes)
    mapping = Vector{Vector{Int}}(undef, num_odes)

    for i in 1:num_odes
        indices = Int[]
        offset = 0
        for var_idx in 1:ode_sizes[i]
            # Count how many ODEs have at least var_idx variables
            num_with_var = count(>=(var_idx), ode_sizes)
            # This ODE's position among those with this variable
            position_in_var = count(>=(var_idx), ode_sizes[1:i])

            offset += (var_idx == 1 ? 0 : count(>=(var_idx-1), ode_sizes))
            push!(indices, offset + position_in_var)
        end
        mapping[i] = indices
    end
    return mapping
end

# Store in LockstepFunction
struct LockstepFunction{O, C}
    # ...
    perindex_mapping::Union{Nothing, Vector{Vector{Int}}}
end

function _get_idxs(lockstep_func::LockstepFunction{PerIndex}, i)
    return lockstep_func.perindex_mapping[i]
end
```

**Note**: PerIndex with variable sizes is complex and may not provide cache benefits. Consider warning users or restricting to PerODE for heterogeneous case.

### Challenge 4: Batching Utilities Update

**Problem**: `batch_initial_conditions` assumes uniform size.

**Solution**: Add overloads for variable sizes:

```julia
# Heterogeneous: vector of vectors with potentially different sizes
function batch_initial_conditions(
    u0::Vector{<:AbstractVector},
    ode_sizes::Vector{Int}
)
    @assert length(u0) == length(ode_sizes)
    for (i, u) in enumerate(u0)
        @assert length(u) == ode_sizes[i] "ODE $i: expected size $(ode_sizes[i]), got $(length(u))"
    end
    return vcat(u0...)
end

# Uniform case (backward compatible)
function batch_initial_conditions(
    u0::AbstractVector{T},
    num_odes::Int,
    ode_size::Int
) where {T <: Number}
    # existing implementation
end
```

### Challenge 5: Solution Extraction

**Problem**: `extract_solutions` uses fixed stride.

**Solution**: Use cumulative offsets:

```julia
function extract_solutions(lockstep_func::LockstepFunction, sol)
    offsets = cumsum([0; lockstep_func.ode_sizes[1:end-1]])

    individual_sols = map(1:lockstep_func.num_odes) do i
        start_idx = offsets[i] + 1
        end_idx = offsets[i] + lockstep_func.ode_sizes[i]

        u_series = [Array(u[start_idx:end_idx]) for u in sol.u]

        return (u = u_series, t = sol.t)
    end
    return individual_sols
end
```

### Challenge 6: Type System Design

**Problem**: Need to store both uniform and heterogeneous data efficiently.

**Solution**: Unified representation with smart constructors:

```julia
struct LockstepFunction{O, C}
    fs::Vector{Function}              # Always vector (even if uniform)
    num_odes::Int
    ode_sizes::Vector{Int}            # Always vector (even if uniform)
    total_size::Int                   # Cached: sum(ode_sizes)
    internal_threading::Bool
    ordering::O
    callbacks::C
    perindex_mapping::Union{Nothing, Vector{Vector{Int}}}  # Only for PerIndex + heterogeneous
end

# Constructor 1: Uniform case (backward compatible API)
function LockstepFunction(
    f::Function,
    ode_size::Int,
    num_odes::Int;
    internal_threading=true,
    ordering=PerODE(),
    callbacks=nothing
)
    fs = fill(f, num_odes)  # Replicate function reference
    ode_sizes = fill(ode_size, num_odes)
    total_size = ode_size * num_odes

    perindex_mapping = ordering isa PerIndex ? nothing : nothing  # All same size, use simple arithmetic

    return LockstepFunction(fs, num_odes, ode_sizes, total_size,
                           internal_threading, ordering, callbacks, perindex_mapping)
end

# Constructor 2: Heterogeneous - different functions, potentially different sizes
function LockstepFunction(
    fs::Vector{<:Function},
    ode_sizes::Vector{Int};
    internal_threading=true,
    ordering=PerODE(),
    callbacks=nothing
)
    num_odes = length(fs)
    @assert length(ode_sizes) == num_odes "Must provide one size per function"
    @assert all(>(0), ode_sizes) "All ODE sizes must be positive"

    total_size = sum(ode_sizes)

    # Compute PerIndex mapping if needed and sizes vary
    perindex_mapping = if ordering isa PerIndex && !allequal(ode_sizes)
        @warn "PerIndex ordering with variable sizes may have poor performance"
        _compute_perindex_mapping(ode_sizes, num_odes)
    else
        nothing
    end

    return LockstepFunction(fs, num_odes, ode_sizes, total_size,
                           internal_threading, ordering, callbacks, perindex_mapping)
end

# Constructor 3: Heterogeneous - different functions, same size
function LockstepFunction(
    fs::Vector{<:Function},
    ode_size::Int;
    kwargs...
)
    return LockstepFunction(fs, fill(ode_size, length(fs)); kwargs...)
end
```

### Challenge 7: Kernel Dispatch Update

**Problem**: Need to call different functions in `ode_kernel!`.

**Solution**: Index into function vector:

```julia
@inline function ode_kernel!(i, lockstep_func::LockstepFunction, du, u, p, t)
    idxs = _get_idxs(lockstep_func, i)
    u_i = view(u, idxs)
    du_i = view(du, idxs)
    p_i = _get_ode_parameters(p, i, lockstep_func.num_odes)

    # Call the i-th function (was: lockstep_func.f)
    lockstep_func.fs[i](du_i, u_i, p_i, t)

    return nothing
end
```

**Performance Note**: `lockstep_func.fs[i]` introduces dynamic dispatch. Acceptable because:
1. Time integration kernel work dominates
2. Vector lookup is fast (pointer + offset)
3. Can optimize with `@inline` and compiler hints

### Challenge 8: Callback System Compatibility

**Problem**: SubIntegrator must work with variable sizes.

**Solution**: Already uses `_get_idxs`, so just works after indexing fix:

```julia
function SubIntegrator(integrator, lockstep_func, ode_idx)
    idxs = _get_idxs(lockstep_func, ode_idx)  # Uses updated _get_idxs
    u_view = view(integrator.u, idxs)
    p_i = _get_ode_parameters(integrator.p, ode_idx, lockstep_func.num_odes)
    return SubIntegrator(integrator, lockstep_func, ode_idx, u_view, p_i)
end
```

**No changes needed** to `create_lockstep_callbacks` - it already works per-ODE.

### Challenge 9: GPU Extension Compatibility

**Problem**: Do GPU extensions need updates?

**Solution**: No changes needed! Extensions already call core `ode_kernel!`:

```julia
# ext/LockstepODECUDAExt.jl (and others)
KA.@kernel function ode_kernel!(lockstep_func, du, u, p, t)
    i = KA.@index(Global)
    LockstepODE.ode_kernel!(i, lockstep_func, du, u, p, t)
end
```

Core `ode_kernel!` handles function dispatch, so extensions transparent to changes.

### Challenge 10: Backward Compatibility

**Problem**: Breaking changes vs. smooth migration.

**Decision**: Allow breaking changes (version 0.1.0) for cleaner design:

**Breaking Changes**:
- Struct field `f` → `fs` (affects direct field access)
- Struct field `ode_size` → `ode_sizes` (affects direct field access)
- Added `total_size` field
- Added `perindex_mapping` field

**Backward Compatible**:
- Constructor API: `LockstepFunction(f, ode_size, num_odes)` still works
- User-facing functions: `batch_initial_conditions`, `extract_solutions` overloaded
- GPU extensions: no changes needed
- Callback system: no changes needed

**Migration Path**:
- Most users don't access struct fields directly
- Constructor signature unchanged for common case
- Tests will catch any direct field access

## Implementation Plan

### Phase 1: Core Changes
1. Update `LockstepFunction` struct with new fields
2. Add three constructor variants (uniform, heterogeneous+sizes, heterogeneous+same_size)
3. Update `ode_kernel!` to index function vector
4. Add validation in constructors

### Phase 2: Indexing Rewrite
1. Update `_get_idxs` for PerODE with variable sizes
2. Implement `_compute_perindex_mapping` helper
3. Update `_get_idxs` for PerIndex with mapping lookup
4. Add helper: `_get_ode_size(lockstep_func, i)`

### Phase 3: Utilities Update
1. Add `batch_initial_conditions` overloads for variable sizes
2. Update `extract_solutions` to use cumulative offsets
3. Add offset computation helper: `_compute_cumulative_offsets`

### Phase 4: Testing
1. Test uniform case (regression - ensure no performance loss)
2. Test heterogeneous same-size (different functions, same dimensions)
3. Test heterogeneous variable-size (different functions, different dimensions)
4. Test both PerODE and PerIndex layouts
5. Test callback system with heterogeneous ODEs
6. Performance benchmarks

### Phase 5: Documentation
1. Update README with heterogeneous example
2. Update docstrings with new constructor signatures
3. Update CLAUDE.md architecture notes
4. Add this design doc to repository

### Phase 6: Release
1. Update Project.toml to version 0.1.0
2. Add CHANGELOG entry documenting breaking changes
3. Tag release

## Example Usage

### Before (Current - Uniform)
```julia
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# 100 Lorenz systems with different parameters
lockstep_func = LockstepFunction(lorenz!, 3, 100)
u0 = batch_initial_conditions([1.0, 0.0, 0.0], 100, 3)
p = [rand(3) for _ in 1:100]  # Different parameters each

prob = ODEProblem(lockstep_func, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5())
```

### After (Heterogeneous)
```julia
# Three different ODE systems
function decay!(du, u, p, t)
    du[1] = -p * u[1]
end

function oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p * u[1]
end

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Combine different systems
fs = [decay!, oscillator!, lorenz!]
sizes = [1, 2, 3]
lockstep_func = LockstepFunction(fs, sizes)

# Initial conditions for each system
u0_decay = [1.0]
u0_osc = [1.0, 0.0]
u0_lorenz = [1.0, 0.0, 0.0]
u0 = batch_initial_conditions([u0_decay, u0_osc, u0_lorenz], sizes)

# Parameters for each system
p = [0.1, 2.0, (10.0, 28.0, 8/3)]

prob = ODEProblem(lockstep_func, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5())

# Extract individual solutions
sols = extract_solutions(lockstep_func, sol)
decay_sol = sols[1]
osc_sol = sols[2]
lorenz_sol = sols[3]
```

## Performance Considerations

### Dynamic Dispatch Overhead
- **Impact**: `fs[i]` lookup + indirect call vs. direct function call
- **Mitigation**: Time integration dominates; function call is ~1-5% overhead
- **Future**: Can use FunctionWrappers.jl for type-stable wrapping if needed

### Memory Layout
- **PerODE**: Same performance as current (contiguous per-ODE)
- **PerIndex with variable sizes**: Poor cache behavior, complex mapping
- **Recommendation**: Encourage PerODE for heterogeneous, PerIndex only for uniform

### GPU Performance
- GPU kernel launch overhead dominates function dispatch
- Same parallel strategy across GPU threads
- Heterogeneous should have minimal impact on GPU performance

### Indexing Cost
- Cumulative offset computation: O(1) with caching
- PerIndex mapping: O(1) lookup into pre-computed vector
- Both negligible vs. ODE evaluation cost

## Risks and Mitigations

### Risk 1: Type Instability Performance Impact
**Mitigation**: Benchmark and profile. If problematic, add FunctionWrappers.jl as optimization.

### Risk 2: PerIndex Complexity with Variable Sizes
**Mitigation**: Add comprehensive tests. Consider deprecating PerIndex for heterogeneous if too complex.

### Risk 3: Breaking Changes User Impact
**Mitigation**: Clear CHANGELOG, migration guide, version bump to 0.1.0, deprecation warnings.

### Risk 4: Testing Coverage
**Mitigation**: Comprehensive test suite covering all combinations (layouts × sizes × callbacks).

### Risk 5: Documentation Clarity
**Mitigation**: Clear examples in README, thorough docstrings, this design doc.

## Success Criteria

1. ✅ Can solve N different ODE functions in lockstep
2. ✅ Supports variable ODE sizes per function
3. ✅ Backward compatible constructor API
4. ✅ Both PerODE and PerIndex layouts work
5. ✅ Callback system works with heterogeneous ODEs
6. ✅ GPU extensions work unchanged
7. ✅ No performance regression for uniform case (<5% overhead)
8. ✅ Comprehensive test coverage (>90%)
9. ✅ Clear documentation with examples

## Future Extensions

1. **Optimize dispatch**: Use FunctionWrappers.jl for type-stable function calls
2. **Sparse coupling**: Allow ODEs to share state variables (coupled systems)
3. **Adaptive sizing**: Allow ODEs to change size during integration (DAE, event-driven)
4. **Compile-time specialization**: Generated function approach for small, fixed heterogeneous sets
5. **Multi-rate integration**: Allow different ODEs to use different timesteps (complex!)

## Conclusion

Adding heterogeneous ODE support is a **moderate complexity feature** requiring:
- ~500 lines of core code changes
- ~300 lines of new tests
- ~150 lines of documentation

**Key Insight**: The architecture is well-designed for extension. Most challenges are addressable with careful indexing and smart constructors. The lockstep property (synchronized time) is preserved naturally through the existing design.

**Recommendation**: Proceed with Vector{Function} approach, allowing breaking changes for cleaner design. The uniform case remains performant, and heterogeneous case enables valuable new use cases.
