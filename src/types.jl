"""
Core type definitions for LockstepODE v2.0

Multi-integrator architecture: N independent ODEs synchronized at fixed intervals.
"""

using SciMLBase: ReturnCode

"""
    LockstepFunction{F, C}

Coordinates solving N independent ODEs with optional synchronization.

# Fields
- `f::F`: ODE function with signature `f(du, u, p, t)`
- `num_odes::Int`: Number of independent ODEs
- `ode_size::Int`: Size of each ODE's state vector
- `sync_interval::Float64`: Time between sync points (0 = no sync)
- `coupling!`: In-place coupling function `coupling!(states, t)` or `nothing`
- `coupling_indices`: Indices to couple, or `nothing` for all indices
- `callbacks::C`: Per-ODE callbacks (nothing, single, or Vector)

# Constructor
```julia
LockstepFunction(f, ode_size, num_odes;
    sync_interval=0.0,
    coupling!=nothing,
    coupling_indices=nothing,
    callbacks=nothing
)
```

# Example
```julia
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

# Mean-field coupling on second component
function couple!(states, t)
    x2_mean = sum(s[2] for s in states) / length(states)
    for s in states
        s[2] = x2_mean
    end
end

lf = LockstepFunction(lorenz!, 3, 10;
    sync_interval=0.1,
    coupling=couple!,
    coupling_indices=[2]
)
```
"""
struct LockstepFunction{F, C}
    f::F
    num_odes::Int
    ode_size::Int
    sync_interval::Float64
    coupling::Union{Nothing, Function}  # In-place coupling function coupling!(states, t)
    coupling_indices::Union{Nothing, Vector{Int}}
    callbacks::C
end

function LockstepFunction(
    f::F,
    ode_size::Integer,
    num_odes::Integer;
    sync_interval::Real=0.0,
    coupling::Union{Nothing, Function}=nothing,
    coupling_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    callbacks::C=nothing
) where {F, C}
    # Validate inputs
    num_odes > 0 || throw(ArgumentError("num_odes must be positive, got $num_odes"))
    ode_size > 0 || throw(ArgumentError("ode_size must be positive, got $ode_size"))
    sync_interval >= 0 || throw(ArgumentError("sync_interval must be non-negative, got $sync_interval"))

    # Validate coupling_indices
    indices = if coupling_indices === nothing
        nothing
    else
        idx_vec = Vector{Int}(coupling_indices)
        for idx in idx_vec
            1 <= idx <= ode_size || throw(ArgumentError(
                "coupling_indices must be in range [1, $ode_size], got $idx"
            ))
        end
        idx_vec
    end

    # Warn if sync_interval set but no coupling
    if sync_interval > 0 && coupling === nothing
        @warn "sync_interval=$sync_interval specified but no coupling function provided"
    end

    return LockstepFunction{F, C}(
        f,
        Int(num_odes),
        Int(ode_size),
        Float64(sync_interval),
        coupling,
        indices,
        callbacks
    )
end

"""
    LockstepSolution{S}

Combined solution from N synchronized ODEs.

# Fields
- `solutions::Vector{S}`: Individual ODESolutions (one per ODE)
- `sync_times::Vector{Float64}`: Times where synchronization occurred
- `retcode::ReturnCode.T`: Overall return code

# Accessors
- `sol[i]`: Get i-th ODE's solution
- `length(sol)`: Number of ODEs
- `sol.sync_times`: Synchronization time points
"""
struct LockstepSolution{S}
    solutions::Vector{S}
    sync_times::Vector{Float64}
    retcode::ReturnCode.T
end
