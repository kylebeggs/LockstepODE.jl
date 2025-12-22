"""
Core type definitions for LockstepODE v2.0

Multi-integrator architecture: N independent ODEs solved in parallel.
Supports both Ensemble mode (N independent integrators) and Batched mode (single batched integrator).
"""

using SciMLBase: ReturnCode, ODEFunction

#==============================================================================#
# Mode and Ordering Types
#==============================================================================#

"""
    LockstepMode

Abstract type for lockstep execution modes.

Subtypes:
- `Ensemble`: N independent ODE integrators, synchronized via lockstep time tracking
- `Batched`: Single ODE integrator with batched state vector, parallel RHS evaluation
"""
abstract type LockstepMode end

"""
    Ensemble <: LockstepMode

N independent ODE integrators mode.

Each ODE has its own integrator with potentially different adaptive timesteps.
Synchronization is via lockstep time tracking (minimum time across all integrators).

Best for:
- Per-ODE control and introspection
- Complex callbacks per ODE
- ModelingToolkit integration
"""
struct Ensemble <: LockstepMode end

"""
    Batched <: LockstepMode

Single batched ODE integrator mode.

All ODEs share a single integrator with one flat state vector.
RHS evaluation is parallelized across ODEs.

Best for:
- Large N (many ODEs)
- GPU acceleration
- Maximum performance when per-ODE control is not needed
"""
struct Batched <: LockstepMode end

"""
    MemoryOrdering

Abstract type for batched mode memory layouts.

Subtypes:
- `PerODE`: Each ODE's state stored contiguously (default)
- `PerIndex`: Same variable index across ODEs stored contiguously
"""
abstract type MemoryOrdering end

"""
    PerODE <: MemoryOrdering

Memory layout where each ODE's state is stored contiguously.

For 2 ODEs with 3 variables each:
`[u1_ode1, u2_ode1, u3_ode1, u1_ode2, u2_ode2, u3_ode2]`

This is generally the best layout for most use cases.
"""
struct PerODE <: MemoryOrdering end

"""
    PerIndex <: MemoryOrdering

Memory layout where same variable index is stored contiguously across ODEs.

For 2 ODEs with 3 variables each:
`[u1_ode1, u1_ode2, u2_ode1, u2_ode2, u3_ode1, u3_ode2]`

May provide better cache locality for operations that access the same
variable across all ODEs.
"""
struct PerIndex <: MemoryOrdering end

#==============================================================================#
# Batched Mode Options
#==============================================================================#

"""
    BatchedOpts{O<:MemoryOrdering}

Options for Batched mode execution.

# Fields
- `ordering::O`: Memory layout (PerODE or PerIndex)
- `internal_threading::Bool`: Enable CPU threading for RHS evaluation
"""
struct BatchedOpts{O<:MemoryOrdering}
    ordering::O
    internal_threading::Bool
end

function BatchedOpts(; ordering::MemoryOrdering=PerODE(), internal_threading::Bool=true)
    return BatchedOpts(ordering, internal_threading)
end

#==============================================================================#
# LockstepFunction
#==============================================================================#

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

#==============================================================================#
# Parameter Extraction
#==============================================================================#

"""
    _get_ode_parameters(p, i::Int, num_odes::Int)

Extract parameters for the i-th ODE from batched parameters.

If `p` is a vector with length == num_odes, returns `p[i]`.
Otherwise returns `p` unchanged (shared parameter or nothing).
"""
@inline function _get_ode_parameters(p::AbstractVector, i::Int, num_odes::Int)
    return length(p) == num_odes ? p[i] : p
end

@inline _get_ode_parameters(p, ::Int, ::Int) = p

#==============================================================================#
# LockstepFunction Callable Interface
#==============================================================================#

"""
    (lf::LockstepFunction)(du, u, p, t)

Callable interface for use with ODEProblem. Operates on flat interleaved state vector.

State layout: [u1_1, u1_2, ..., u1_M, u2_1, u2_2, ..., u2_M, ..., uN_1, ..., uN_M]
where N = num_odes and M = ode_size.

Parameters `p` should be a vector of per-ODE parameters (length == num_odes),
or a single value/nothing to share across all ODEs.
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

        # Call the underlying ODE function with per-ODE parameters
        p_i = _get_ode_parameters(p, i, N)
        lf.f(du_i, u_i, p_i, t)
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
