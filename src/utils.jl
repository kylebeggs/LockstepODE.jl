# Parameter handling functions
function _get_ode_parameters(p::AbstractVector, i::Int, num_odes::Int)
    length(p) == num_odes ? p[i] : p
end
_get_ode_parameters(p, ::Int, ::Int) = p

# Callback handling functions
function _get_ode_callback(callbacks::AbstractVector, i::Int, num_odes::Int)
    length(callbacks) == num_odes ? callbacks[i] : callbacks
end
_get_ode_callback(callbacks, ::Int, ::Int) = callbacks
_get_ode_callback(::Nothing, ::Int, ::Int) = nothing

# Index management functions for LockstepFunction
function _get_idxs(lockstep_func::LockstepFunction{PerODE, F, C}, i) where {F, C}
    ((i - 1) * lockstep_func.ode_size + 1):(i * lockstep_func.ode_size)
end

function _get_idxs(lockstep_func::LockstepFunction{PerIndex, F, C}, i) where {F, C}
    i:(lockstep_func.num_odes):((lockstep_func.ode_size * (lockstep_func.num_odes - 1)) + i)
end

# Batching and extraction utilities

"""
    batch_initial_conditions(u0::AbstractVector{T}, num_odes::Int, ode_size::Int) where {T <: Number}
    batch_initial_conditions(u0::AbstractVector{<:AbstractVector}, num_odes::Int, ode_size::Int)

Prepare initial conditions for batch ODE solving with LockstepFunction.

This function handles three cases:
1. If `u0` is already properly batched (length = `num_odes * ode_size`), returns it unchanged
2. If `u0` represents a single ODE (length = `ode_size`), replicates it for all ODEs
3. If `u0` is a vector of vectors (one per ODE), concatenates them into a single vector

# Arguments
- `u0`: Initial conditions - either a single vector or vector of vectors
- `num_odes::Int`: Number of ODE systems to solve
- `ode_size::Int`: Number of state variables per ODE

# Returns
A single flattened vector containing all initial conditions

# Examples
```julia
# Case 1: Single initial condition replicated for all ODEs
u0_single = [1.0, 2.0, 3.0]  # 3 variables
u0_batched = batch_initial_conditions(u0_single, 10, 3)
# Returns: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, ...] (length 30)

# Case 2: Different initial conditions for each ODE
u0_multiple = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3 ODEs, 2 variables each
u0_batched = batch_initial_conditions(u0_multiple, 3, 2)
# Returns: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Case 3: Already batched
u0_prebatched = ones(20)  # 10 ODEs × 2 variables
u0_batched = batch_initial_conditions(u0_prebatched, 10, 2)
# Returns: unchanged u0_prebatched
```
"""
function batch_initial_conditions(
        u0::AbstractVector{T}, num_odes::Int, ode_size::Int
) where {T <: Number}
    expected_batched_size = num_odes * ode_size

    # If u0 is already the correct batched size, return as-is
    if length(u0) == expected_batched_size
        return u0
    elseif length(u0) == ode_size
        # If u0 is a single ODE, repeat it for all ODEs
        result = repeat(u0, num_odes)
        return result
    else
        error("u0 length ($(length(u0))) doesn't match expected single ODE size ($ode_size) or batched size ($expected_batched_size)")
    end
end

function batch_initial_conditions(
        u0::AbstractVector{<:AbstractVector}, num_odes::Int, ::Int
)
    @assert length(u0)==num_odes "Length of u0 vector must equal num_odes"
    result = vcat(u0...)
    return result
end

"""
    extract_solutions(lockstep_func::LockstepFunction{O, F}, sol) where {O, F}

Extract individual ODE solutions from a batched solution.

After solving a batched ODE problem with `LockstepFunction`, this function separates 
the combined solution back into individual solutions for each ODE system.

# Arguments
- `lockstep_func::LockstepFunction`: The LockstepFunction used to create the ODE problem
- `sol`: The solution object returned by `solve()`

# Returns
A vector of named tuples, where each tuple contains:
- `u`: Time series of state vectors for one ODE system
- `t`: Time points (shared across all ODEs)

# Example
```julia
# Setup and solve batched ODEs
function exponential_decay!(du, u, p, t)
    du[1] = -p * u[1]
end

lockstep_func = LockstepFunction(exponential_decay!, 1, 5)
u0_batched = [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 ODEs with different initial values
prob = ODEProblem(lockstep_func, u0_batched, (0.0, 10.0), 0.1)
sol = solve(prob, Tsit5())

# Extract individual solutions
individual_solutions = extract_solutions(lockstep_func, sol)

# Access the solution for the 3rd ODE
ode3_solution = individual_solutions[3]
ode3_states = ode3_solution.u  # Vector of state values over time
ode3_times = ode3_solution.t    # Time points
```
"""
function extract_solutions(lockstep_func::LockstepFunction{O, F, C}, sol) where {O, F, C}
    individual_sols = map(1:(lockstep_func.num_odes)) do i
        start_idx = (i - 1) * lockstep_func.ode_size + 1
        end_idx = i * lockstep_func.ode_size

        u_series = [Array(u[start_idx:end_idx]) for u in sol.u]

        return (u = u_series, t = sol.t)
    end
    return individual_sols
end
