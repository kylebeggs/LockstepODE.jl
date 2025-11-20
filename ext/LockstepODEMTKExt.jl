module LockstepODEMTKExt

using LockstepODE
using ModelingToolkit
using ModelingToolkit: ODESystem, structural_simplify, ODEFunction
import LockstepODE: AbstractODEWrapper, LockstepFunction, SimpleWrapper, param_size
import OrdinaryDiffEq: ODEProblem

"""
    MTKWrapper{F,U,P,S}

Wrapper for ModelingToolkit.jl ODESystem objects.

This wrapper handles the conversion of symbolic ODESystem objects into
compiled numerical functions that can be efficiently executed in parallel
kernels (both CPU and GPU).

# Type Parameters
- `F`: Type of the compiled ODE function
- `U`: Type information for state variables
- `P`: Type information for parameters
- `S`: Type of the ODESystem

# Fields
- `compiled_func::F`: The compiled numerical function from ODEFunction
- `ode_size::Int`: Number of state variables in the system
- `param_size::Int`: Number of parameters in the system
- `system::S`: The simplified ODESystem (for parameter mapping)

# Thread Safety
The compiled function is designed to be thread-safe and can be called
from multiple threads simultaneously with different state views.
"""
struct MTKWrapper{F,U,P,S} <: AbstractODEWrapper
    compiled_func::F
    ode_size::Int
    param_size::Int
    state_type::Type{U}
    param_type::Type{P}
    system::S
end

"""
    MTKWrapper(sys::ODESystem)

Construct an MTKWrapper from a ModelingToolkit ODESystem.

This constructor automatically performs structural simplification and
extracts a compiled numerical function suitable for parallel execution.

# Arguments
- `sys::ODESystem`: The ModelingToolkit ODE system to wrap

# Returns
An `MTKWrapper` containing the compiled function and metadata

# Example
```julia
using ModelingToolkit

@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [
    D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z
]

@named sys = ODESystem(eqs, t)
wrapper = MTKWrapper(sys)

# Use with LockstepFunction
lockstep_func = LockstepFunction(wrapper, 3, 100)
```
"""
function MTKWrapper(sys::ODESystem)
    # Perform structural simplification
    sys_simplified = structural_simplify(sys)

    # Get the number of states and parameters
    n_states = length(unknowns(sys_simplified))
    n_params = length(parameters(sys_simplified))

    # Create ODEFunction which generates optimized code
    ode_func = ODEFunction(sys_simplified)

    # Extract the compiled function
    # For MTK, the function is in ode_func.f field
    # This is the actual function that implements (du, u, p, t)
    compiled_func = ode_func.f

    # Determine types for type stability
    state_type = Vector{Float64}
    param_type = Any

    return MTKWrapper{typeof(compiled_func), state_type, param_type, typeof(sys_simplified)}(
        compiled_func, n_states, n_params, state_type, param_type, sys_simplified
    )
end

"""
    (wrapper::MTKWrapper)(du, u, p, t)

Call the wrapped ModelingToolkit function.

This method is optimized for parallel execution and works with array views.
The compiled MTK function is called directly with the provided state views.

# Arguments
- `du`: Derivative array (or view)
- `u`: State array (or view)
- `p`: Parameters (will be converted to vector if needed)
- `t`: Time

# Note
The wrapper ensures the compiled function is compatible with both threaded
execution on CPU and GPU kernel execution. MTK functions typically expect
parameters as vectors, so scalar parameters are wrapped in a vector.
"""
@inline function (wrapper::MTKWrapper)(du, u, p, t)
    # MTK functions often expect parameters as vectors/arrays
    # Convert scalar parameters to single-element vectors
    p_converted = p isa Number ? [p] : p
    wrapper.compiled_func(du, u, p_converted, t)
    return nothing
end

"""
    param_size(wrapper::MTKWrapper)

Get the number of parameters in the ModelingToolkit system.
"""
param_size(wrapper::MTKWrapper) = wrapper.param_size

"""
    transform_parameters(wrapper::MTKWrapper, params::Vector{Dict})
    transform_parameters(wrapper::MTKWrapper, params::Vector{<:AbstractVector{<:Pair}})

Transform symbolic parameter specifications into a flat vector matching MTK's canonical ordering.

This function takes per-ODE parameter dictionaries or pair vectors and converts them
to a single flattened vector where parameters are ordered according to MTK's internal
canonical ordering (obtained via `parameters(system)`).

# Arguments
- `wrapper::MTKWrapper`: The MTK wrapper containing the system and parameter information
- `params`: Vector of parameter specifications, one per ODE. Each element can be:
  - `Dict{Symbol, <:Real}`: Dictionary mapping parameter symbols to values
  - `Vector{Pair{Symbol, <:Real}}`: Vector of parameter symbol => value pairs

# Returns
- `Vector{Float64}`: Flattened parameter vector in canonical order

# Example
```julia
@parameters α β γ δ
# ... define system ...
wrapper = MTKWrapper(sys)

# User provides params in any order
params = [
    Dict(α=>1.5, β=>1.0, γ=>3.0, δ=>2.0),
    Dict(α=>1.0, β=>2.0, γ=>2.0, δ=>1.5)
]

# Transform to canonical order
flat_params = transform_parameters(wrapper, params)  # [1.5, 1.0, 2.0, 3.0, 1.0, 2.0, 1.5, 2.0]
```
"""
function transform_parameters(
    wrapper::MTKWrapper,
    params::Vector{<:Union{<:AbstractDict, <:AbstractVector{<:Pair}}}
)
    canonical_params = parameters(wrapper.system)
    flat_params = Float64[]

    for (ode_idx, param_spec) in enumerate(params)
        # Convert to Dict for uniform handling
        param_dict = param_spec isa AbstractDict ? param_spec : Dict(param_spec)

        # Extract parameters in canonical order
        for p_sym in canonical_params
            if !haskey(param_dict, p_sym)
                error("Missing parameter $p_sym for ODE $ode_idx. " *
                      "Expected parameters: $(canonical_params)")
            end
            push!(flat_params, Float64(param_dict[p_sym]))
        end
    end

    return flat_params
end

"""
    LockstepFunction(sys::ODESystem, ode_size::Int, num_odes::Int;
                     internal_threading=true, ordering=PerODE(), callbacks=nothing)

Construct a LockstepFunction directly from a ModelingToolkit ODESystem.

This constructor automatically wraps the ODESystem in an MTKWrapper, handling
all necessary compilation and simplification steps.

# Arguments
- `sys::ODESystem`: The ModelingToolkit ODE system
- `ode_size::Int`: Number of state variables (should match system size)
- `num_odes::Int`: Number of parallel ODE instances to solve

# Keyword Arguments
- `internal_threading::Bool = true`: Enable internal threading for parallel execution
- `ordering = PerODE()`: Memory ordering (`PerODE` or `PerIndex`)
- `callbacks = nothing`: Optional callbacks to apply to ODEs

# Returns
A `LockstepFunction` instance ready for use with `ODEProblem`

# Example
```julia
using ModelingToolkit, LockstepODE

# Define a simple exponential decay system
@parameters α
@variables t x(t)
D = Differential(t)

eqs = [D(x) ~ -α * x]
@named decay_sys = ODESystem(eqs, t)

# Create LockstepFunction for 100 parallel instances
lockstep_func = LockstepFunction(decay_sys, 1, 100)

# Use with standard OrdinaryDiffEq workflow
using OrdinaryDiffEq
u0 = ones(100)
tspan = (0.0, 10.0)
p = 0.1
prob = ODEProblem(lockstep_func, u0, tspan, p)
sol = solve(prob, Tsit5())
```
"""
function LockstepFunction(
        sys::ODESystem,
        ode_size::Int,
        num_odes::Int;
        internal_threading = true,
        ordering = LockstepODE.PerODE(),
        callbacks = nothing
)
    # Create MTK wrapper
    wrapper = MTKWrapper(sys)

    # Verify size matches
    if wrapper.ode_size != ode_size
        @warn "Provided ode_size ($ode_size) does not match system size ($(wrapper.ode_size)). Using system size."
        ode_size = wrapper.ode_size
    end

    # Construct LockstepFunction with wrapper
    return LockstepFunction(wrapper, num_odes, ode_size, internal_threading, ordering, callbacks)
end

"""
    ODEProblem(f::LockstepFunction{O, <:MTKWrapper}, u0, tspan,
               params::Vector{<:Union{Dict, AbstractVector{<:Pair}}}; kwargs...)

Construct an ODEProblem with symbolic parameter specification for MTK systems.

This constructor allows users to specify parameters using dictionaries or vectors of pairs,
where parameter symbols map to values. The parameters are automatically transformed to
match ModelingToolkit's internal canonical ordering.

# Arguments
- `f::LockstepFunction`: LockstepFunction wrapping an MTK system
- `u0`: Initial conditions (batched format)
- `tspan`: Time span tuple (tstart, tend)
- `params`: Vector of parameter specifications, one per ODE. Each can be:
  - `Dict{Symbol, <:Real}`: Mapping parameter symbols to values
  - `Vector{Pair{Symbol, <:Real}}`: Pairs of parameter symbols and values

# Keyword Arguments
- `kwargs...`: Additional ODEProblem keyword arguments (callback, etc.)

# Example
```julia
using LockstepODE, ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D

# Define Lotka-Volterra system
@parameters α β γ δ
@variables x(t) y(t)
eqs = [D(x) ~ α*x - β*x*y, D(y) ~ δ*x*y - γ*y]
@named lotka_volterra = ODESystem(eqs, t)

# Create lockstep function
lockstep_func = LockstepFunction(lotka_volterra, 2, 3)

# Specify parameters symbolically - order doesn't matter!
params = [
    Dict(α=>1.5, β=>1.0, γ=>3.0, δ=>2.0),  # ODE 1
    Dict(α=>1.0, β=>2.0, γ=>2.0, δ=>1.5),  # ODE 2
    Dict(α=>1.0, β=>1.0, γ=>1.0, δ=>2.0)   # ODE 3
]

u0 = vcat([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
prob = ODEProblem(lockstep_func, u0, (0.0, 10.0), params)
sol = solve(prob, Tsit5())
```

# Notes
- Parameters must be provided in order (one Dict/Vector per ODE)
- All parameters defined in the system must be present in each specification
- Flat vector parameters (standard API) still work without any changes
"""
function ODEProblem(
    f::LockstepFunction{O, <:MTKWrapper, C},
    u0, tspan,
    params::Vector{<:Union{<:AbstractDict, <:AbstractVector{<:Pair}}};
    kwargs...
) where {O, C}
    # Transform symbolic parameters to flat canonical-ordered vector
    params_flat = transform_parameters(f.wrapper, params)

    # Call base ODEProblem constructor with flattened parameters
    return invoke(ODEProblem,
                  Tuple{LockstepFunction, Any, Any, Any},
                  f, u0, tspan, params_flat; kwargs...)
end

end # module
