# ModelingToolkit Parameter Ordering Issue & Solution

## The Problem

**Issue:** ModelingToolkit internally reorders parameters, causing parameter mismatches when batching multiple ODEs.

**Example:**
```julia
@parameters α β γ δ
eqs = [D(x) ~ α*x - β*x*y, D(y) ~ δ*x*y - γ*y]

# User defines: [α, β, γ, δ]
# MTK stores as: [α, β, δ, γ]  ← δ and γ swapped!
```

**Impact on LockstepODE:**
- Users must provide parameters for multiple ODEs as flat vectors
- They must manually discover MTK's canonical ordering via `parameters(structural_simplify(sys))`
- Error-prone and unintuitive

## Our Solution

### 1. Fixed Parameter Slicing (Core Issue)

**Added `param_size` to `MTKWrapper`:**
```julia
struct MTKWrapper{F,U,P,S}
    compiled_func::F
    ode_size::Int
    param_size::Int  # NEW: track number of parameters
    system::S        # NEW: store ODESystem for transformation
end
```

**Updated `_get_ode_parameters()` to handle batched parameters:**
```julia
function _get_ode_parameters(p::AbstractVector, i::Int, num_odes::Int, param_size::Int)
    if length(p) == num_odes
        return p[i]  # Scalar per ODE
    elseif param_size > 0 && length(p) == num_odes * param_size
        # Batched: slice out this ODE's parameters
        start_idx = (i - 1) * param_size + 1
        return view(p, start_idx:start_idx + param_size - 1)
    else
        return p  # Shared parameters
    end
end
```

### 2. Symbolic Parameter API (User-Facing Fix)

**Added `ODEProblem` constructor accepting `Vector{Dict}` or `Vector{Vector{Pair}}`:**
```julia
function ODEProblem(
    f::LockstepFunction{O, <:MTKWrapper, C},
    u0, tspan,
    params::Vector{<:Union{<:AbstractDict, <:AbstractVector{<:Pair}}};
    kwargs...
)
    # Transform symbolic params to flat canonical-ordered vector
    params_flat = transform_parameters(f.wrapper, params)
    # Delegate to base constructor
    return invoke(ODEProblem, Tuple{LockstepFunction, Any, Any, Any},
                  f, u0, tspan, params_flat; kwargs...)
end
```

**Transformation logic:**
```julia
function transform_parameters(wrapper::MTKWrapper, params::Vector)
    canonical_params = parameters(wrapper.system)  # MTK's order
    flat_params = Float64[]

    for param_dict in params
        for p_sym in canonical_params
            push!(flat_params, param_dict[p_sym])  # Extract in canonical order
        end
    end

    return flat_params
end
```

## User Experience

**Before:**
```julia
# Must manually check MTK ordering
sys_simplified = structural_simplify(lotka_volterra)
canonical_order = parameters(sys_simplified)  # [α, β, δ, γ]

# Must provide in canonical order
params = vcat([1.5, 1.0, 1.0, 3.0], [1.0, 2.0, 1.5, 2.0], ...)
prob = ODEProblem(lockstep_func, u0, tspan, params)
```

**After:**
```julia
# Specify symbolically - any order works!
params = [
    Dict(α=>1.5, β=>1.0, γ=>3.0, δ=>1.0),
    Dict(α=>1.0, β=>2.0, γ=>2.0, δ=>1.5),
    ...
]
prob = ODEProblem(lockstep_func, u0, tspan, params)
```

## Questions for MTK Expert

1. **Is storing the simplified `ODESystem` in the wrapper problematic?** (Memory/invalidation concerns?)

2. **Better way to handle parameter ordering?** Should we use MTK's `MTKParameters` type directly instead of flattening?

3. **Any MTK utilities we missed?** Is there an existing API for parameter name → canonical index mapping?

4. **Extension design concerns?** We're extending `ODEProblem` in a package extension - any issues with this approach?

5. **Future-proofing?** Will MTK's parameter handling change in ways that break this approach?

## Implementation Stats

- **Files changed:** 4 (extension, utils, tests, examples)
- **Lines added:** ~200
- **Tests:** 93 passing (11 new MTK tests)
- **Overhead:** Zero runtime cost (transformation at problem creation)
- **Compatibility:** Fully backward compatible
