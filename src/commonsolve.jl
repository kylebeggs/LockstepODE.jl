"""
CommonSolve.jl interface implementations for LockstepODE.

Provides init, solve, and solve! methods that integrate with the SciML ecosystem.
Dispatches on mode: LockstepProblem{Ensemble} or LockstepProblem{Batched}.
"""

using OrdinaryDiffEq: ODEProblem, ODESolution, CallbackSet
using OrdinaryDiffEq: init as ode_init, solve as ode_solve, solve! as ode_solve!
using SciMLBase: ReturnCode
using OhMyThreads: tforeach
import CommonSolve: solve!, init, solve

#==============================================================================#
# Ensemble Mode: init
#==============================================================================#

"""
    init(prob::LockstepProblem{Ensemble}, alg; kwargs...)

Initialize an EnsembleLockstepIntegrator for Ensemble mode.

Creates N independent ODE integrators that can be stepped manually or solved to completion.

# Example
```julia
prob = LockstepProblem{Ensemble}(lf, u0s, (0.0, 10.0), p)  # Explicit Ensemble mode
integ = init(prob, Tsit5())
step!(integ)
step!(integ, 0.1, true)
sol = solve!(integ)
```
"""
function CommonSolve.init(
    prob::LockstepProblem{Ensemble, LF, U, T, P, Opts},
    alg::A;
    save_everystep::Bool = true,
    kwargs...
) where {LF, U, T, P, Opts, A}
    lf = prob.lf

    # Create N individual ODEProblems
    problems = _create_individual_problems(lf, prob.u0s, prob.tspan, prob.ps)

    # Initialize first integrator to infer concrete type (avoids Vector{Any})
    first_integ = ode_init(problems[1], alg; save_everystep, kwargs...)
    I = typeof(first_integ)

    integrators = Vector{I}(undef, lf.num_odes)
    integrators[1] = first_integ

    # Initialize remaining integrators in parallel
    tforeach(2:lf.num_odes) do i
        integrators[i] = ode_init(problems[i], alg; save_everystep, kwargs...)
    end

    t0, tf = prob.tspan

    return EnsembleLockstepIntegrator{A, I, LF, T}(
        integrators,
        lf,
        alg,
        t0,
        prob.tspan,
        sign(tf - t0),
        ReturnCode.Default
    )
end

#==============================================================================#
# Ensemble Mode: solve
#==============================================================================#

"""
    solve(prob::LockstepProblem{Ensemble}, alg; kwargs...)

Solve the Ensemble mode problem to completion.

# Example
```julia
prob = LockstepProblem(lf, u0s, (0.0, 10.0), p)
sol = solve(prob, Tsit5())
```
"""
function CommonSolve.solve(
    prob::LockstepProblem{Ensemble},
    alg;
    kwargs...
)::LockstepSolution
    integ = init(prob, alg; kwargs...)
    return solve!(integ)
end

#==============================================================================#
# Ensemble Mode: solve!
#==============================================================================#

"""
    solve!(integ::EnsembleLockstepIntegrator)

Solve the Ensemble integrator to completion.

Solves all N ODE integrators in parallel independently.

# Example
```julia
integ = init(prob, Tsit5())
step!(integ)  # Optional manual stepping
sol = solve!(integ)  # Complete the solve
```
"""
function CommonSolve.solve!(integ::EnsembleLockstepIntegrator)::LockstepSolution
    # Solve all integrators to completion in parallel
    tforeach(eachindex(integ.integrators)) do i
        ode_solve!(integ.integrators[i])
    end
    integ.t = integ.tspan[2]

    # Update retcode after completion
    update_retcode!(integ)

    return _finalize_ensemble_solution(integ)
end

function _finalize_ensemble_solution(integ::EnsembleLockstepIntegrator)::LockstepSolution
    solutions = [sub.sol for sub in integ.integrators]

    retcode = ReturnCode.Success
    for sol in solutions
        if sol.retcode != ReturnCode.Success && sol.retcode != ReturnCode.Default
            retcode = sol.retcode
            break
        end
    end

    return LockstepSolution(solutions, retcode)
end

#==============================================================================#
# Batched Mode: init
#==============================================================================#

"""
    init(prob::LockstepProblem{Batched}, alg; kwargs...)

Initialize a BatchedLockstepIntegrator for Batched mode.

Creates a single ODE integrator with batched state vector. RHS evaluation
is parallelized across ODEs. Supports GPU arrays.

# Example
```julia
prob = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), p)
integ = init(prob, Tsit5())
step!(integ)
sol = solve!(integ)
```
"""
function CommonSolve.init(
    prob::LockstepProblem{Batched, LF, U, T, P, Opts},
    alg;
    save_everystep::Bool = true,
    kwargs...
) where {LF, U, T, P, Opts}
    lf = prob.lf
    opts = prob.opts

    # Create BatchedFunction for RHS evaluation (stores params internally)
    bf = BatchedFunction(lf, opts, prob.ps)

    # Batch initial conditions with correct ordering
    u0_batched = batch_u0s(prob.u0s, opts.ordering)

    # Create and wrap callbacks
    cb = create_lockstep_callbacks(bf)

    # Merge with any user-provided callback
    user_cb = get(kwargs, :callback, nothing)
    merged_cb = if !isnothing(cb) && !isnothing(user_cb)
        CallbackSet(cb, user_cb)
    elseif !isnothing(cb)
        cb
    else
        user_cb
    end

    # Create single batched ODEProblem
    # Pass nothing for p - params are stored in bf (bypasses SciMLBase introspection)
    ode_prob = ODEProblem(bf, u0_batched, prob.tspan, nothing; callback=merged_cb)

    # Initialize the underlying integrator
    raw_integ = ode_init(ode_prob, alg; save_everystep, kwargs...)

    return BatchedLockstepIntegrator(raw_integ, lf, bf, opts)
end

#==============================================================================#
# Batched Mode: solve
#==============================================================================#

"""
    solve(prob::LockstepProblem{Batched}, alg; kwargs...)

Solve the Batched mode problem to completion.

# Example
```julia
prob = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), p)
sol = solve(prob, Tsit5())
```
"""
function CommonSolve.solve(
    prob::LockstepProblem{Batched},
    alg;
    kwargs...
)::LockstepSolution
    integ = init(prob, alg; kwargs...)
    return solve!(integ)
end

#==============================================================================#
# Batched Mode: solve!
#==============================================================================#

"""
    solve!(integ::BatchedLockstepIntegrator)

Solve the Batched integrator to completion.

# Example
```julia
integ = init(prob, Tsit5())
step!(integ)  # Optional manual stepping
sol = solve!(integ)  # Complete the solve
```
"""
function CommonSolve.solve!(integ::BatchedLockstepIntegrator)::LockstepSolution
    # Solve the underlying integrator
    ode_solve!(integ.integrator)

    return _finalize_batched_solution(integ)
end

function _finalize_batched_solution(integ::BatchedLockstepIntegrator)::LockstepSolution
    raw_sol = integ.integrator.sol
    bf = integ.bf
    lf = integ.lf

    # Extract individual solutions as BatchedSubSolution wrappers
    solutions = [BatchedSubSolution(raw_sol, bf, i) for i in 1:lf.num_odes]

    return LockstepSolution(solutions, raw_sol.retcode)
end

#==============================================================================#
# Helper: create individual ODEProblems (Ensemble mode only)
#==============================================================================#

function _create_individual_problems(
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
            problems[i] = ODEProblem(lf.f, u0_i, tspan, p_i; callback = cb_i)
        end
    end

    return problems
end
