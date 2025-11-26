"""
Core type definitions for LockstepODE v2.0

Multi-integrator architecture: N independent ODEs solved in parallel.
"""

using SciMLBase: ReturnCode, ODEFunction

"""
    LockstepFunction{F, C}

Coordinates solving N independent ODEs in parallel.

# Fields
- `f::F`: ODE function with signature `f(du, u, p, t)`
- `num_odes::Int`: Number of independent ODEs
- `ode_size::Int`: Size of each ODE's state vector
- `callbacks::C`: Per-ODE callbacks (nothing, single, or Vector)

# Constructor
```julia
LockstepFunction(f, ode_size, num_odes; callbacks=nothing)
```

# Example
```julia
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

lf = LockstepFunction(lorenz!, 3, 10)
```
"""
struct LockstepFunction{F, C}
    f::F
    num_odes::Int
    ode_size::Int
    callbacks::C
end

function LockstepFunction(
    f::F,
    ode_size::Integer,
    num_odes::Integer;
    callbacks::C=nothing
) where {F, C}
    num_odes > 0 || throw(ArgumentError("num_odes must be positive, got $num_odes"))
    ode_size > 0 || throw(ArgumentError("ode_size must be positive, got $ode_size"))

    return LockstepFunction{F, C}(f, Int(num_odes), Int(ode_size), callbacks)
end

"""
    (lf::LockstepFunction)(du, u, p, t)

Callable interface for use with ODEProblem. Operates on flat interleaved state vector.

State layout: [u1_1, u1_2, ..., u1_M, u2_1, u2_2, ..., u2_M, ..., uN_1, ..., uN_M]
where N = num_odes and M = ode_size.

Parameters `p` should be a vector of per-ODE parameters (length == num_odes).
"""
function (lf::LockstepFunction)(du, u, p, t)
    N = lf.num_odes
    M = lf.ode_size

    # Process each ODE in parallel
    Threads.@threads for i in 1:N
        # Calculate index range for this ODE
        idx_start = (i - 1) * M + 1
        idx_end = i * M

        # Create views into the flat arrays
        du_i = @view du[idx_start:idx_end]
        u_i = @view u[idx_start:idx_end]

        # Call the underlying ODE function
        lf.f(du_i, u_i, p[i], t)
    end

    return nothing
end

"""
    LockstepSolution{S}

Combined solution from N independent ODEs.

# Fields
- `solutions::Vector{S}`: Individual ODESolutions (one per ODE)
- `retcode::ReturnCode.T`: Overall return code

# Accessors
- `sol[i]`: Get i-th ODE's solution
- `length(sol)`: Number of ODEs
"""
struct LockstepSolution{S}
    solutions::Vector{S}
    retcode::ReturnCode.T
end
