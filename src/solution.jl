"""
Solution accessors and utilities for LockstepODE v2.0
"""

# ============================================================================
# LockstepSolution accessors
# ============================================================================

"""
    getindex(sol::LockstepSolution, i::Int)

Access the i-th ODE's solution.

# Example
```julia
sol = solve(prob, Tsit5())
first_ode_sol = sol[1]
second_ode_at_t5 = sol[2](5.0)  # interpolation
```
"""
Base.getindex(sol::LockstepSolution, i::Int) = sol.solutions[i]

"""
    length(sol::LockstepSolution)

Number of ODEs in the solution.
"""
Base.length(sol::LockstepSolution) = length(sol.solutions)

"""
    iterate(sol::LockstepSolution)

Iterate over individual ODE solutions.

# Example
```julia
for ode_sol in sol
    plot!(ode_sol.t, ode_sol[1,:])
end
```
"""
Base.iterate(sol::LockstepSolution) = iterate(sol.solutions)
Base.iterate(sol::LockstepSolution, state) = iterate(sol.solutions, state)

"""
    firstindex(sol::LockstepSolution)
"""
Base.firstindex(sol::LockstepSolution) = 1

"""
    lastindex(sol::LockstepSolution)
"""
Base.lastindex(sol::LockstepSolution) = length(sol.solutions)

"""
    eachindex(sol::LockstepSolution)
"""
Base.eachindex(sol::LockstepSolution) = eachindex(sol.solutions)

# ============================================================================
# Extraction utilities
# ============================================================================

"""
    extract_solutions(lf::LockstepFunction, sol::LockstepSolution)

Extract individual ODE solutions as a vector.

This is a convenience function for compatibility with the v1 API.
With LockstepSolution, you can also just use `sol.solutions` or iterate directly.

# Returns
Vector of individual ODESolution objects.

# Example
```julia
lf = LockstepFunction(f!, 3, 10)
prob = LockstepProblem(lf, u0s, tspan, p)
sol = solve(prob, Tsit5())

individual_sols = extract_solutions(lf, sol)
for (i, s) in enumerate(individual_sols)
    println("ODE \$i final state: \$(s.u[end])")
end
```
"""
function extract_solutions(::LockstepFunction, sol::LockstepSolution)
    return sol.solutions
end

"""
    extract_at_time(sol::LockstepSolution, t::Real)

Extract all ODE states at a specific time (with interpolation).

# Returns
Vector of state vectors, one per ODE.

# Example
```julia
states_at_5 = extract_at_time(sol, 5.0)
for (i, state) in enumerate(states_at_5)
    println("ODE \$i at t=5: \$state")
end
```
"""
function extract_at_time(sol::LockstepSolution, t::Real)
    return [s(t) for s in sol.solutions]
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, sol::LockstepSolution)
    n = length(sol.solutions)
    print(io, "LockstepSolution with $n ODEs ($(sol.retcode))")
end

function Base.show(io::IO, ::MIME"text/plain", sol::LockstepSolution)
    n = length(sol.solutions)
    println(io, "LockstepSolution")
    println(io, "  ODEs: $n")
    print(io, "  Return code: $(sol.retcode)")
end
