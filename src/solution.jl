"""
Solution accessors and utilities for LockstepODE v2.0

Supports both Ensemble mode (vector of ODESolutions) and Batched mode
(vector of BatchedSubSolution wrappers with interpolation support).
"""

#==============================================================================#
# BatchedSubSolution - Interpolatable wrapper for Batched mode
#==============================================================================#

"""
    BatchedSubSolution{S, BF}

Wrapper providing per-ODE solution access with interpolation for Batched mode.

When `sol = solve(prob, alg)` is called in Batched mode, `sol[i]` returns a
`BatchedSubSolution` that provides:
- `sol[i](t)`: Interpolate state at time `t`
- `sol[i].u`: Time series of states (lazy extraction)
- `sol[i].t`: Time points

# Fields
- `parent::S`: The full batched ODESolution
- `bf::BF`: BatchedFunction for index calculations
- `ode_idx::Int`: Which ODE this represents
"""
struct BatchedSubSolution{S, BF}
    parent::S
    bf::BF
    ode_idx::Int
end

# Interpolation: sol(t) returns state at time t
function (sol::BatchedSubSolution)(t::Real)
    idxs = _get_idxs(sol.bf, sol.ode_idx)
    return sol.parent(t)[idxs]
end

# Property accessors
function Base.getproperty(sol::BatchedSubSolution, sym::Symbol)
    if sym === :parent || sym === :bf || sym === :ode_idx
        return getfield(sol, sym)
    elseif sym === :t
        # Return time points from parent solution
        return getfield(sol, :parent).t
    elseif sym === :u
        # Extract time series for this ODE
        parent = getfield(sol, :parent)
        bf = getfield(sol, :bf)
        idx = getfield(sol, :ode_idx)
        idxs = _get_idxs(bf, idx)
        return [Array(u[idxs]) for u in parent.u]
    elseif sym === :retcode
        return getfield(sol, :parent).retcode
    else
        return getproperty(getfield(sol, :parent), sym)
    end
end

function Base.propertynames(::BatchedSubSolution)
    return (:parent, :bf, :ode_idx, :t, :u, :retcode)
end

# Indexing into state at timepoint: sol[i] returns state at i-th timepoint
function Base.getindex(sol::BatchedSubSolution, i::Int)
    idxs = _get_idxs(sol.bf, sol.ode_idx)
    return sol.parent.u[i][idxs]
end

# Range indexing for multiple timepoints
function Base.getindex(sol::BatchedSubSolution, r::AbstractRange)
    idxs = _get_idxs(sol.bf, sol.ode_idx)
    return [sol.parent.u[i][idxs] for i in r]
end

# Length = number of saved timepoints
Base.length(sol::BatchedSubSolution) = length(sol.parent.u)
Base.firstindex(::BatchedSubSolution) = 1
Base.lastindex(sol::BatchedSubSolution) = length(sol.parent.u)
Base.eachindex(sol::BatchedSubSolution) = 1:length(sol.parent.u)

# Iteration over timepoints
function Base.iterate(sol::BatchedSubSolution)
    return length(sol.parent.u) > 0 ? (sol[1], 1) : nothing
end

function Base.iterate(sol::BatchedSubSolution, state::Int)
    next = state + 1
    return next <= length(sol.parent.u) ? (sol[next], next) : nothing
end

# Display
function Base.show(io::IO, sol::BatchedSubSolution)
    n = length(sol.parent.u)
    print(io, "BatchedSubSolution(ODE $(sol.ode_idx), $n timepoints)")
end

#==============================================================================#
# LockstepSolution accessors
#==============================================================================#

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
