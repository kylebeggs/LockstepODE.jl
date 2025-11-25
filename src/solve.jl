"""
Solve methods for LockstepODE v2.0

Multi-integrator solving with optional fixed-interval synchronization.
"""

using OrdinaryDiffEq: ODEProblem, ODESolution, init, step!
using OrdinaryDiffEq: solve as ode_solve
using SciMLBase: ReturnCode

"""
    solve(lf::LockstepFunction, u0s, tspan, p, alg; kwargs...)

Solve N independent ODEs with optional fixed-interval synchronization.

# Arguments
- `lf::LockstepFunction`: The lockstep function coordinator
- `u0s`: Initial conditions - Vector of vectors (one per ODE) or single vector (replicated)
- `tspan`: Time span tuple `(t0, tf)`
- `p`: Parameters - `nothing`, shared params, or Vector of per-ODE params
- `alg`: ODE solver algorithm (e.g., `Tsit5()`)

# Keyword Arguments
- `save_everystep::Bool=true`: Save solution at every step
- All other kwargs passed to OrdinaryDiffEq.solve

# Returns
- `LockstepSolution`: Contains individual solutions and sync times

# Example
```julia
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

lf = LockstepFunction(lorenz!, 3, 10)
u0s = [[1.0, 0.0, 0.0] for _ in 1:10]
sol = solve(lf, u0s, (0.0, 10.0), (10.0, 28.0, 8/3), Tsit5())
```
"""
function solve(
    lf::LockstepFunction,
    u0s,
    tspan::Tuple,
    p,
    alg;
    kwargs...
)
    # Normalize inputs
    u0s_normalized = _normalize_u0s(u0s, lf.num_odes, lf.ode_size)
    ps_normalized = _normalize_params(p, lf.num_odes)

    # Determine solve mode
    if lf.sync_interval <= 0 || lf.coupling === nothing
        return solve_independent(lf, u0s_normalized, tspan, ps_normalized, alg; kwargs...)
    else
        return solve_synchronized(lf, u0s_normalized, tspan, ps_normalized, alg; kwargs...)
    end
end

# Convenience: allow omitting parameters
function solve(lf::LockstepFunction, u0s, tspan::Tuple, alg; kwargs...)
    return solve(lf, u0s, tspan, nothing, alg; kwargs...)
end

"""
Solve N ODEs independently in parallel (no synchronization).
"""
function solve_independent(
    lf::LockstepFunction,
    u0s::Vector,
    tspan::Tuple,
    ps::Vector,
    alg;
    kwargs...
)
    # Create individual problems
    problems = create_individual_problems(lf, u0s, tspan, ps)

    # Solve in parallel
    solutions = Vector{Any}(undef, lf.num_odes)
    Threads.@threads for i in 1:lf.num_odes
        solutions[i] = ode_solve(problems[i], alg; kwargs...)
    end

    return LockstepSolution(solutions, Float64[], ReturnCode.Success)
end

"""
Solve N ODEs with fixed-interval synchronization and coupling.
"""
function solve_synchronized(
    lf::LockstepFunction,
    u0s::Vector,
    tspan::Tuple,
    ps::Vector,
    alg;
    save_everystep::Bool=true,
    kwargs...
)
    t0, tf = tspan
    Δt = lf.sync_interval

    # Create individual problems
    problems = create_individual_problems(lf, u0s, tspan, ps)

    # Initialize integrators
    integrators = [init(p, alg; save_everystep=save_everystep, kwargs...)
                   for p in problems]

    sync_times = Float64[t0]
    t = t0

    while t < tf
        t_next = min(t + Δt, tf)

        # Step all integrators to next sync point in parallel
        step_to_sync!(integrators, t_next)

        # Apply coupling at sync point (not at final time)
        if lf.coupling !== nothing && t_next < tf
            apply_coupling!(integrators, lf.coupling, lf.coupling_indices)
        end

        push!(sync_times, t_next)
        t = t_next
    end

    # Build final solutions
    solutions = [integ.sol for integ in integrators]

    # Determine overall return code
    retcode = ReturnCode.Success
    for sol in solutions
        if sol.retcode != ReturnCode.Success && sol.retcode != ReturnCode.Default
            retcode = sol.retcode
            break
        end
    end

    return LockstepSolution(solutions, sync_times, retcode)
end

"""
Advance all integrators to target time in parallel.
Each integrator uses its own adaptive timestepping.
"""
function step_to_sync!(integrators, target_time::Real)
    Threads.@threads for integ in integrators
        while integ.t < target_time
            remaining = target_time - integ.t
            if integ.dt > remaining
                # Take a smaller step to land exactly on target
                step!(integ, remaining, true)
            else
                step!(integ)
            end

            # Check for failure
            if integ.sol.retcode != ReturnCode.Default && integ.sol.retcode != ReturnCode.Success
                break
            end
        end
    end
end

"""
Apply coupling function to all integrator states.
"""
function apply_coupling!(integrators, coupling::Function, coupling_indices)
    # Collect current states (views into integrator.u)
    states = [integ.u for integ in integrators]
    t = integrators[1].t

    # Call user's coupling function (modifies states in-place)
    coupling(states, t)

    # If coupling_indices specified, we could validate here
    # but the coupling! function is responsible for which indices it modifies
end

"""
Create N individual ODEProblems.
"""
function create_individual_problems(
    lf::LockstepFunction,
    u0s::Vector,
    tspan::Tuple,
    ps::Vector
)
    callbacks_normalized = _normalize_callbacks(lf.callbacks, lf.num_odes)

    problems = Vector{ODEProblem}(undef, lf.num_odes)
    for i in 1:lf.num_odes
        u0_i = u0s[i]
        p_i = ps[i]
        cb_i = callbacks_normalized[i]

        if cb_i === nothing
            problems[i] = ODEProblem(lf.f, u0_i, tspan, p_i)
        else
            problems[i] = ODEProblem(lf.f, u0_i, tspan, p_i; callback=cb_i)
        end
    end

    return problems
end

# ============================================================================
# Input normalization helpers
# ============================================================================

"""
Normalize initial conditions to Vector of Vectors.
"""
function _normalize_u0s(u0s::AbstractVector{<:AbstractVector}, num_odes::Int, ode_size::Int)
    length(u0s) == num_odes || throw(ArgumentError(
        "Expected $num_odes initial conditions, got $(length(u0s))"
    ))
    for (i, u0) in enumerate(u0s)
        length(u0) == ode_size || throw(ArgumentError(
            "ODE $i: expected state size $ode_size, got $(length(u0))"
        ))
    end
    return u0s
end

function _normalize_u0s(u0::AbstractVector{<:Number}, num_odes::Int, ode_size::Int)
    # Single u0 - replicate for all ODEs
    length(u0) == ode_size || throw(ArgumentError(
        "Expected state size $ode_size, got $(length(u0))"
    ))
    return [copy(u0) for _ in 1:num_odes]
end

"""
Normalize parameters to Vector.

Parameter normalization rules:
- `nothing` → `[nothing, nothing, ...]`
- Scalar (e.g., `0.5`) → `[0.5, 0.5, ...]` (shared)
- Vector of scalars with length == num_odes (e.g., `[0.1, 0.5, 1.0]`) → per-ODE scalar params
- Vector of non-scalars (e.g., `[[1,2], [3,4]]`) with length == num_odes → per-ODE param sets
- Tuple (e.g., `(1.0, 2.0)`) → `[(1.0, 2.0), (1.0, 2.0), ...]` (shared)
"""
function _normalize_params(p::Nothing, num_odes::Int)
    return fill(nothing, num_odes)
end

function _normalize_params(p::AbstractVector{<:Number}, num_odes::Int)
    if length(p) == num_odes
        # Per-ODE scalar parameters: [0.5, 1.0, 2.0] → each ODE gets one value
        return p
    else
        # Shared vector parameter: [1.0, 2.0] (not matching num_odes) → replicate
        return fill(p, num_odes)
    end
end

function _normalize_params(p::AbstractVector, num_odes::Int)
    # Non-numeric element type (vectors, tuples, etc.)
    if length(p) == num_odes
        # Per-ODE parameter sets
        return p
    else
        # Shared parameters
        return fill(p, num_odes)
    end
end

function _normalize_params(p, num_odes::Int)
    # Scalar or tuple - shared across all ODEs
    return fill(p, num_odes)
end

"""
Normalize callbacks to Vector.
"""
function _normalize_callbacks(callbacks::Nothing, num_odes::Int)
    return fill(nothing, num_odes)
end

function _normalize_callbacks(callbacks::AbstractVector, num_odes::Int)
    length(callbacks) == num_odes || throw(ArgumentError(
        "Expected $num_odes callbacks, got $(length(callbacks))"
    ))
    return callbacks
end

function _normalize_callbacks(callback, num_odes::Int)
    # Single callback - share across all ODEs
    return fill(callback, num_odes)
end
