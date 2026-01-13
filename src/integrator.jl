"""
LockstepIntegrator types for CommonSolve.jl interface.

Two modes:
- EnsembleLockstepIntegrator: N individual OrdinaryDiffEq integrators (Ensemble mode)
- BatchedLockstepIntegrator: Single batched OrdinaryDiffEq integrator (Batched mode)
"""

using SciMLBase: ReturnCode
using SciMLBase: step! as sciml_step!, reinit! as sciml_reinit!
import SciMLBase: step!, reinit!
using OhMyThreads: tforeach

#==============================================================================#
# Abstract Type
#==============================================================================#

"""
    AbstractLockstepIntegrator{M}

Abstract type for lockstep integrators, parametric on mode.
"""
abstract type AbstractLockstepIntegrator{M<:LockstepMode} end

# Type alias for backward compatibility
const LockstepIntegrator = AbstractLockstepIntegrator

#==============================================================================#
# Ensemble Mode Integrator
#==============================================================================#

"""
    EnsembleLockstepIntegrator{A, I, LF, T}

Unified integrator wrapping N individual OrdinaryDiffEq integrators.
Implements CommonSolve step!/solve!/reinit! interface.

# Fields
- `integrators`: Vector of N individual ODE integrators
- `lf`: Reference to LockstepFunction
- `alg`: ODE algorithm
- `t`: Current lockstep time (minimum of all integrator times)
- `tspan`: Original time span
- `tdir`: Time direction (+1 or -1)
- `retcode`: Overall return code

# Accessors
- `integ[i]`: Returns i-th underlying OrdinaryDiffEq integrator
- `integ.u`: Returns vector of current states `[integ[1].u, ...]`
- `integ.t`: Returns current lockstep time
- `length(integ)`: Number of ODEs
"""
mutable struct EnsembleLockstepIntegrator{A, I, LF, T} <: AbstractLockstepIntegrator{Ensemble}
    integrators::Vector{I}
    lf::LF
    alg::A
    t::T
    tspan::Tuple{T, T}
    tdir::Int
    retcode::ReturnCode.T
end

#==============================================================================#
# Batched Mode Integrator
#==============================================================================#

"""
    BatchedLockstepIntegrator{I, LF, BF, Opts}

Wrapper around a single OrdinaryDiffEq integrator with batched state.
Provides uniform interface for Batched mode solving.

# Fields
- `integrator`: Single OrdinaryDiffEq integrator with batched state
- `lf`: Reference to LockstepFunction
- `bf`: BatchedFunction used for RHS evaluation
- `opts`: BatchedOpts configuration

# Accessors
- `integ[i]`: Returns SubIntegrator for i-th ODE
- `integ.u`: Returns vector of state views `[view(u, idxs_1), ...]`
- `integ.t`: Returns current time
- `length(integ)`: Number of ODEs
"""
mutable struct BatchedLockstepIntegrator{I, LF, BF, Opts} <: AbstractLockstepIntegrator{Batched}
    integrator::I
    lf::LF
    bf::BF
    opts::Opts
end

#==============================================================================#
# Ensemble Mode Property Accessors
#==============================================================================#

function Base.getproperty(integ::EnsembleLockstepIntegrator, sym::Symbol)
    if sym === :u
        return [sub.u for sub in getfield(integ, :integrators)]
    elseif sym === :p
        return [sub.p for sub in getfield(integ, :integrators)]
    elseif sym === :dt
        return _compute_minimum_dt(getfield(integ, :integrators))
    else
        return getfield(integ, sym)
    end
end

@inline function _compute_minimum_dt(integrators)
    dt_min = integrators[1].dt
    @inbounds @simd for i in 2:length(integrators)
        dt_i = integrators[i].dt
        dt_min = ifelse(dt_i < dt_min, dt_i, dt_min)
    end
    return dt_min
end

@inline function _compute_minimum_t(integrators)
    t_min = integrators[1].t
    @inbounds @simd for i in 2:length(integrators)
        t_i = integrators[i].t
        t_min = ifelse(t_i < t_min, t_i, t_min)
    end
    return t_min
end

function Base.propertynames(::EnsembleLockstepIntegrator)
    return (:integrators, :lf, :alg, :t, :tspan, :tdir, :retcode, :u, :p, :dt)
end

# Indexing
Base.getindex(integ::EnsembleLockstepIntegrator, i::Int) = integ.integrators[i]
Base.length(integ::EnsembleLockstepIntegrator) = length(integ.integrators)
Base.eachindex(integ::EnsembleLockstepIntegrator) = eachindex(integ.integrators)
Base.firstindex(::EnsembleLockstepIntegrator) = 1
Base.lastindex(integ::EnsembleLockstepIntegrator) = length(integ.integrators)

# Iteration
Base.iterate(integ::EnsembleLockstepIntegrator) = iterate(integ.integrators)
Base.iterate(integ::EnsembleLockstepIntegrator, state) = iterate(integ.integrators, state)

#==============================================================================#
# Batched Mode Property Accessors
#==============================================================================#

function Base.getproperty(integ::BatchedLockstepIntegrator, sym::Symbol)
    if sym === :u
        # Return vector of views for each ODE
        bf = getfield(integ, :bf)
        raw_u = getfield(integ, :integrator).u
        return [view(raw_u, _get_idxs(bf, i)) for i in 1:bf.lf.num_odes]
    elseif sym === :p
        # Return per-ODE parameters (stored in bf)
        bf = getfield(integ, :bf)
        return bf.params
    elseif sym === :t
        return getfield(integ, :integrator).t
    elseif sym === :dt
        return getfield(integ, :integrator).dt
    elseif sym === :tspan
        return getfield(integ, :integrator).sol.prob.tspan
    elseif sym === :tdir
        return sign(getfield(integ, :integrator).sol.prob.tspan[2] -
                    getfield(integ, :integrator).sol.prob.tspan[1])
    elseif sym === :retcode
        return getfield(integ, :integrator).sol.retcode
    else
        return getfield(integ, sym)
    end
end

function Base.propertynames(::BatchedLockstepIntegrator)
    return (:integrator, :lf, :bf, :opts, :u, :p, :t, :dt, :tspan, :tdir, :retcode)
end

# Indexing - returns SubIntegrator for per-ODE access
Base.getindex(integ::BatchedLockstepIntegrator, i::Int) =
    SubIntegrator(integ.integrator, integ.bf, i)
Base.length(integ::BatchedLockstepIntegrator) = integ.lf.num_odes
Base.eachindex(integ::BatchedLockstepIntegrator) = 1:integ.lf.num_odes
Base.firstindex(::BatchedLockstepIntegrator) = 1
Base.lastindex(integ::BatchedLockstepIntegrator) = integ.lf.num_odes

# Iteration - iterates over SubIntegrators
function Base.iterate(integ::BatchedLockstepIntegrator)
    return integ.lf.num_odes > 0 ? (integ[1], 1) : nothing
end

function Base.iterate(integ::BatchedLockstepIntegrator, state::Int)
    next_state = state + 1
    return next_state <= integ.lf.num_odes ? (integ[next_state], next_state) : nothing
end

#==============================================================================#
# Ensemble Mode step! implementations
#==============================================================================#

"""
    step!(integ::EnsembleLockstepIntegrator)

Advance all integrators by one adaptive step.
Updates `integ.t` to the minimum time across all integrators.

Note: Retcode is updated lazily via `update_retcode!` for performance.
Call `update_retcode!(integ)` explicitly if immediate failure detection is needed.
"""
function step!(integ::EnsembleLockstepIntegrator)
    tforeach(eachindex(integ.integrators)) do i
        sciml_step!(integ.integrators[i])
    end

    # Update lockstep time to minimum across all integrators
    integ.t = _compute_minimum_t(integ.integrators)

    return nothing
end

"""
    step!(integ::EnsembleLockstepIntegrator, dt, stop_at_tdt::Bool=false)

Advance all integrators by time `dt`.

# Arguments
- `dt`: Time increment
- `stop_at_tdt`: If `true`, step exactly to `t + dt`. If `false`, step may overshoot.

Note: Retcode is updated lazily via `update_retcode!` for performance.
"""
function step!(integ::EnsembleLockstepIntegrator, dt, stop_at_tdt::Bool = false)
    tforeach(eachindex(integ.integrators)) do i
        sciml_step!(integ.integrators[i], dt, stop_at_tdt)
    end

    integ.t = _compute_minimum_t(integ.integrators)

    return nothing
end

"""
    update_retcode!(integ::EnsembleLockstepIntegrator)

Update the integrator's retcode by checking all sub-integrators for failures.

Called automatically by `solve!` after integration completes.
Call manually during stepping if immediate failure detection is needed.
"""
function update_retcode!(integ::EnsembleLockstepIntegrator)
    # Check all integrators for failures (don't stop at first)
    # The first failure found becomes the "official" retcode
    for sub in integ.integrators
        retcode = sub.sol.retcode
        if retcode != ReturnCode.Default && retcode != ReturnCode.Success
            if integ.retcode == ReturnCode.Default
                integ.retcode = retcode
            end
            # Continue checking all integrators instead of returning early
        end
    end
    return nothing
end

#==============================================================================#
# Batched Mode step! implementations
#==============================================================================#

"""
    step!(integ::BatchedLockstepIntegrator)

Advance the batched integrator by one step.
"""
function step!(integ::BatchedLockstepIntegrator)
    sciml_step!(integ.integrator)
    return nothing
end

"""
    step!(integ::BatchedLockstepIntegrator, dt, stop_at_tdt::Bool=false)

Advance the batched integrator by time `dt`.
"""
function step!(integ::BatchedLockstepIntegrator, dt, stop_at_tdt::Bool = false)
    sciml_step!(integ.integrator, dt, stop_at_tdt)
    return nothing
end

#==============================================================================#
# Ensemble Mode reinit! implementation
#==============================================================================#

"""
    reinit!(integ::EnsembleLockstepIntegrator, u0s=nothing; t0=nothing, tf=nothing, erase_sol=true, kwargs...)

Reinitialize the Ensemble integrator, optionally with new initial conditions.

# Arguments
- `u0s`: New initial conditions (vector of vectors, single vector, or `nothing` to keep current)
- `t0`: New initial time (default: original t0)
- `tf`: New final time (default: original tf)
- `erase_sol`: Whether to erase accumulated solution data
"""
function reinit!(
    integ::EnsembleLockstepIntegrator{A, I, LF, T},
    u0s = nothing;
    t0 = nothing,
    tf = nothing,
    erase_sol::Bool = true,
    kwargs...
) where {A, I, LF, T}
    lf = integ.lf
    new_t0 = t0 === nothing ? integ.tspan[1] : T(t0)
    new_tf = tf === nothing ? integ.tspan[2] : T(tf)

    # Normalize new initial conditions if provided
    u0s_normalized = if u0s !== nothing
        _normalize_u0s(u0s, lf.num_odes, lf.ode_size)
    else
        nothing
    end

    # Reinit each underlying integrator
    for (i, sub_integ) in enumerate(integ.integrators)
        if u0s_normalized === nothing
            # Use original u0 from integrator's problem
            u0_i = sub_integ.sol.prob.u0
        else
            u0_i = u0s_normalized[i]
        end
        sciml_reinit!(sub_integ, u0_i; t0 = new_t0, tf = new_tf, erase_sol, kwargs...)
    end

    # Reset lockstep state
    integ.t = new_t0
    integ.tspan = (new_t0, new_tf)
    integ.tdir = sign(new_tf - new_t0)
    integ.retcode = ReturnCode.Default

    return nothing
end

#==============================================================================#
# Batched Mode reinit! implementation
#==============================================================================#

"""
    reinit!(integ::BatchedLockstepIntegrator, u0s=nothing; t0=nothing, tf=nothing, erase_sol=true, kwargs...)

Reinitialize the Batched integrator, optionally with new initial conditions.

# Arguments
- `u0s`: New initial conditions (vector of vectors, single vector, or `nothing` to keep current)
- `t0`: New initial time (default: original t0)
- `tf`: New final time (default: original tf)
- `erase_sol`: Whether to erase accumulated solution data
"""
function reinit!(
    integ::BatchedLockstepIntegrator,
    u0s = nothing;
    t0 = nothing,
    tf = nothing,
    erase_sol::Bool = true,
    kwargs...
)
    lf = integ.lf
    raw_integ = integ.integrator

    # Get current tspan from raw integrator
    current_tspan = raw_integ.sol.prob.tspan
    T = eltype(current_tspan)
    new_t0 = t0 === nothing ? current_tspan[1] : T(t0)
    new_tf = tf === nothing ? current_tspan[2] : T(tf)

    # Create batched initial conditions with correct ordering
    u0_batched = if u0s !== nothing
        u0s_normalized = _normalize_u0s(u0s, lf.num_odes, lf.ode_size)
        batch_u0s(u0s_normalized, integ.opts.ordering)
    else
        # Use original u0 from problem
        raw_integ.sol.prob.u0
    end

    # Reinit the underlying integrator
    sciml_reinit!(raw_integ, u0_batched; t0 = new_t0, tf = new_tf, erase_sol, kwargs...)

    return nothing
end

