"""
LockstepIntegrator type for CommonSolve.jl interface.

Unified integrator wrapping N individual OrdinaryDiffEq integrators.
"""

using SciMLBase: ReturnCode
using SciMLBase: step! as sciml_step!, reinit! as sciml_reinit!
import SciMLBase: step!, reinit!

"""
    LockstepIntegrator{A, I, LF, T}

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
mutable struct LockstepIntegrator{A, I, LF, T}
    integrators::Vector{I}
    lf::LF
    alg::A
    t::T
    tspan::Tuple{T, T}
    tdir::Int
    retcode::ReturnCode.T
end

# ============================================================================
# Property accessors
# ============================================================================

function Base.getproperty(integ::LockstepIntegrator, sym::Symbol)
    if sym === :u
        return [sub.u for sub in getfield(integ, :integrators)]
    elseif sym === :p
        return [sub.p for sub in getfield(integ, :integrators)]
    elseif sym === :dt
        return minimum(sub.dt for sub in getfield(integ, :integrators))
    else
        return getfield(integ, sym)
    end
end

function Base.propertynames(::LockstepIntegrator)
    return (:integrators, :lf, :alg, :t, :tspan, :tdir, :retcode, :u, :p, :dt)
end

# ============================================================================
# Indexing - returns underlying integrators directly
# ============================================================================

Base.getindex(integ::LockstepIntegrator, i::Int) = integ.integrators[i]
Base.length(integ::LockstepIntegrator) = length(integ.integrators)
Base.eachindex(integ::LockstepIntegrator) = eachindex(integ.integrators)
Base.firstindex(integ::LockstepIntegrator) = 1
Base.lastindex(integ::LockstepIntegrator) = length(integ.integrators)

# ============================================================================
# Iteration over underlying integrators
# ============================================================================

Base.iterate(integ::LockstepIntegrator) = iterate(integ.integrators)
Base.iterate(integ::LockstepIntegrator, state) = iterate(integ.integrators, state)

# ============================================================================
# step! implementations
# ============================================================================

"""
    step!(integ::LockstepIntegrator)

Advance all integrators by one adaptive step.
Updates `integ.t` to the minimum time across all integrators.
"""
function step!(integ::LockstepIntegrator)
    Threads.@threads for i in eachindex(integ.integrators)
        sciml_step!(integ.integrators[i])
    end

    # Update lockstep time to minimum across all integrators
    integ.t = minimum(sub.t for sub in integ.integrators)

    # Check for failures
    _update_retcode!(integ)

    return nothing
end

"""
    step!(integ::LockstepIntegrator, dt, stop_at_tdt::Bool=false)

Advance all integrators by time `dt`.

# Arguments
- `dt`: Time increment
- `stop_at_tdt`: If `true`, step exactly to `t + dt`. If `false`, step may overshoot.
"""
function step!(integ::LockstepIntegrator, dt, stop_at_tdt::Bool = false)
    Threads.@threads for i in eachindex(integ.integrators)
        sciml_step!(integ.integrators[i], dt, stop_at_tdt)
    end

    integ.t = minimum(sub.t for sub in integ.integrators)
    _update_retcode!(integ)

    return nothing
end

function _update_retcode!(integ::LockstepIntegrator)
    for sub in integ.integrators
        if sub.sol.retcode != ReturnCode.Default && sub.sol.retcode != ReturnCode.Success
            integ.retcode = sub.sol.retcode
            return
        end
    end
end

# ============================================================================
# reinit! implementation
# ============================================================================

"""
    reinit!(integ::LockstepIntegrator, u0s=nothing; t0=nothing, tf=nothing, erase_sol=true, kwargs...)

Reinitialize the integrator, optionally with new initial conditions.

# Arguments
- `u0s`: New initial conditions (vector of vectors, single vector, or `nothing` to keep current)
- `t0`: New initial time (default: original t0)
- `tf`: New final time (default: original tf)
- `erase_sol`: Whether to erase accumulated solution data
"""
function reinit!(
    integ::LockstepIntegrator{A, I, LF, T},
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

