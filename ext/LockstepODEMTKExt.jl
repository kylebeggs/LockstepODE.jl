"""
ModelingToolkit extension for LockstepODE v2.0

With the multi-integrator architecture, MTK integration is trivial:
each ODE gets its own standard ODEProblem, so getu/getp/callbacks work natively.
"""
module LockstepODEMTKExt

using LockstepODE
using ModelingToolkit
using ModelingToolkit: ODESystem, ODEFunction, parameters, unknowns, hasdefault, getdefault

import LockstepODE: LockstepFunction

"""
    LockstepFunction(sys::ODESystem, num_odes::Int; kwargs...)

Construct a LockstepFunction from a ModelingToolkit ODESystem.

With the v2.0 multi-integrator architecture, each ODE instance gets its own
standard OrdinaryDiffEq integrator, so MTK features (getu, getp, callbacks)
work natively without special wrappers.

# Arguments
- `sys::ODESystem`: The ModelingToolkit ODE system
- `num_odes::Int`: Number of parallel ODE instances to solve

# Keyword Arguments
- `callbacks=nothing`: Per-ODE callbacks (single or Vector)

# Example
```julia
using ModelingToolkit, LockstepODE, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ=10.0 ρ=28.0 β=8/3
@variables x(t) y(t) z(t)

eqs = [
    D(x) ~ σ*(y - x),
    D(y) ~ x*(ρ - z) - y,
    D(z) ~ x*y - β*z
]

@named lorenz = ODESystem(eqs, t)

lf = LockstepFunction(lorenz, 10)

# Use standard MTK accessors in callbacks
get_x = getu(lorenz, x)
cb = DiscreteCallback(
    (u, t, integrator) -> get_x(integrator) > 20.0,
    integrator -> (integrator.u[1] = 0.0)
)

lf_with_cb = LockstepFunction(lorenz, 10; callbacks=cb)
```
"""
function LockstepFunction(
        sys::ODESystem,
        num_odes::Integer;
        callbacks = nothing
    )
    # Get the compiled ODE function
    ode_func = ODEFunction(sys)
    f = ode_func.f

    # Get system size
    ode_size = length(unknowns(sys))

    # Create LockstepFunction with the compiled MTK function
    return LockstepFunction(f, ode_size, num_odes; callbacks = callbacks)
end

"""
    transform_parameters(sys::ODESystem, params::Vector{<:AbstractDict})

Transform symbolic parameter specifications to flat vectors for per-ODE parameters.

This is a utility function for users who want to specify different parameters
for each ODE using symbolic names.

# Arguments
- `sys::ODESystem`: The MTK system (for canonical parameter ordering)
- `params::Vector{Dict}`: Per-ODE parameter dictionaries

# Returns
Vector of parameter vectors, one per ODE.

# Example
```julia
@parameters α β
@variables x(t)
# ... define system ...

params = [
    Dict(α => 1.0, β => 2.0),  # ODE 1
    Dict(α => 1.5, β => 2.5),  # ODE 2
]

ps = transform_parameters(sys, params)
# ps[1] = [1.0, 2.0] (in canonical order)
# ps[2] = [1.5, 2.5]
```
"""
function transform_parameters(
        sys::ODESystem,
        params::Vector{<:Union{<:AbstractDict, <:AbstractVector{<:Pair}}}
    )
    canonical_params = parameters(sys)

    # Build defaults dictionary from individual parameter metadata (MTK v11 API)
    defaults_dict = Dict{Any, Any}()
    for p in canonical_params
        if hasdefault(p)
            defaults_dict[p] = getdefault(p)
        end
    end

    transformed = Vector{Vector{Float64}}(undef, length(params))

    for (i, param_spec) in enumerate(params)
        param_dict = param_spec isa AbstractDict ? param_spec : Dict(param_spec)
        param_vec = Float64[]

        for p_sym in canonical_params
            if haskey(param_dict, p_sym)
                push!(param_vec, Float64(param_dict[p_sym]))
            elseif haskey(defaults_dict, p_sym)
                push!(param_vec, Float64(defaults_dict[p_sym]))
            else
                error("Missing parameter $p_sym for ODE $i and no default defined")
            end
        end

        transformed[i] = param_vec
    end

    return transformed
end

# Export the transform function
export transform_parameters

end # module
