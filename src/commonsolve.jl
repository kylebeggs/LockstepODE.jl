"""
CommonSolve.jl interface implementations for LockstepODE.

Provides __init, __solve, and solve! methods that integrate with the SciML ecosystem.
"""

using OrdinaryDiffEq: ODEProblem, ODESolution
using OrdinaryDiffEq: init as ode_init, solve as ode_solve
using SciMLBase: ReturnCode
import CommonSolve: solve!, init, solve

# ============================================================================
# init (via SciMLBase.__init)
# ============================================================================

"""
    init(prob::LockstepProblem, alg; kwargs...)

Initialize a LockstepIntegrator for the given problem and algorithm.

Returns a `LockstepIntegrator` that can be stepped manually or solved to completion.

# Example
```julia
prob = LockstepProblem(lf, u0s, (0.0, 10.0), p)
integ = init(prob, Tsit5())
step!(integ)
step!(integ, 0.1, true)
sol = solve!(integ)
```
"""
function CommonSolve.init(
    prob::LockstepProblem{LF, U, T, P},
    alg::A;
    save_everystep::Bool = true,
    kwargs...
) where {LF, U, T, P, A}
    lf = prob.lf

    # Create N individual ODEProblems
    problems = _create_individual_problems(lf, prob.u0s, prob.tspan, prob.ps)

    # Initialize N integrators in parallel
    integrators = Vector{Any}(undef, lf.num_odes)
    Threads.@threads for i in 1:lf.num_odes
        integrators[i] = ode_init(problems[i], alg; save_everystep, kwargs...)
    end

    # Keep as Vector{Any} to handle heterogeneous callback types
    # Each integrator may have different type due to unique callback closures
    I = Any

    t0, tf = prob.tspan

    return LockstepIntegrator{A, I, LF, T}(
        integrators,
        lf,
        alg,
        t0,
        prob.tspan,
        sign(tf - t0),
        ReturnCode.Default
    )
end

# ============================================================================
# solve (via SciMLBase.__solve)
# ============================================================================

"""
    solve(prob::LockstepProblem, alg; kwargs...)

Solve the lockstep problem to completion.

# Example
```julia
prob = LockstepProblem(lf, u0s, (0.0, 10.0), p)
sol = solve(prob, Tsit5())
```
"""
function CommonSolve.solve(
    prob::LockstepProblem,
    alg;
    kwargs...
)::LockstepSolution
    integ = init(prob, alg; kwargs...)
    return solve!(integ)
end

# ============================================================================
# solve!
# ============================================================================

"""
    solve!(integ::LockstepIntegrator)

Solve the integrator to completion and return the solution.

Solves all ODEs in parallel independently.

# Example
```julia
integ = init(prob, Tsit5())
step!(integ)  # Optional manual stepping
sol = solve!(integ)  # Complete the solve
```
"""
function CommonSolve.solve!(integ::LockstepIntegrator)::LockstepSolution
    # Solve all integrators to completion in parallel
    Threads.@threads for i in eachindex(integ.integrators)
        ode_solve!(integ.integrators[i])
    end
    integ.t = integ.tspan[2]

    return _finalize_solution(integ)
end

"""
Build final LockstepSolution from integrator state.
"""
function _finalize_solution(integ::LockstepIntegrator)::LockstepSolution
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

# ============================================================================
# Helper: create individual ODEProblems
# ============================================================================

"""
Create N individual ODEProblems.
"""
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

# Import solve! from OrdinaryDiffEq for internal use
using OrdinaryDiffEq: solve! as ode_solve!
