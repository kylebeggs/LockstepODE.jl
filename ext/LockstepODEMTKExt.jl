"""
ModelingToolkit extension for LockstepODE v2.0

With the multi-integrator architecture, MTK integration is trivial:
each ODE gets its own standard ODEProblem, so getu/getp/callbacks work natively.
"""
module LockstepODEMTKExt

using LockstepODE
using ModelingToolkit
using ModelingToolkit: ODESystem, ODEFunction, parameters, unknowns

# Handle API changes: structural_simplify → mtkcompile (MTK v10+)
const simplify_system = if isdefined(ModelingToolkit, :mtkcompile)
    ModelingToolkit.mtkcompile
else
    ModelingToolkit.structural_simplify
end
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
- `sync_interval::Real=0.0`: Time between sync points (0 = no sync)
- `coupling!::Function=nothing`: In-place coupling function `coupling!(states, t)`
- `coupling_indices::Vector{Int}=nothing`: Which indices to couple
- `callbacks=nothing`: Per-ODE callbacks (single or Vector)
- `simplify::Bool=true`: Whether to structurally simplify the system

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

# Create LockstepFunction - no special wrapper needed!
lf = LockstepFunction(lorenz, 10; sync_interval=0.1)

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
    sync_interval::Real=0.0,
    coupling::Union{Nothing, Function}=nothing,
    coupling_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    callbacks=nothing,
    simplify::Bool=true
)
    # Simplify system if requested
    sys_to_use = simplify ? simplify_system(sys) : sys

    # Get the compiled ODE function
    ode_func = ODEFunction(sys_to_use)
    f = ode_func.f

    # Get system size
    ode_size = length(unknowns(sys_to_use))

    # Create LockstepFunction with the compiled MTK function
    return LockstepFunction(
        f,
        ode_size,
        num_odes;
        sync_interval=sync_interval,
        coupling=coupling,
        coupling_indices=coupling_indices,
        callbacks=callbacks
    )
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
    defaults_dict = ModelingToolkit.defaults(sys)

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
