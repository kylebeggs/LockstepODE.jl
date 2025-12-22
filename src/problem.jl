"""
LockstepProblem type for CommonSolve.jl interface.

Holds all data needed to construct N individual ODEProblems internally.
Parametric on mode: `LockstepProblem{Ensemble}` or `LockstepProblem{Batched}`.
"""

"""
    LockstepProblem{M<:LockstepMode, LF, U, T, P, Opts}

Problem type for batched ODE solving with mode-specific behavior.

# Type Parameters
- `M<:LockstepMode`: Execution mode (Ensemble or Batched)
- `LF`: LockstepFunction type
- `U`: Initial condition element type
- `T`: Time type
- `P`: Parameter element type
- `Opts`: Mode-specific options type (Nothing for Ensemble, BatchedOpts for Batched)

# Fields
- `lf::LF`: LockstepFunction coordinator
- `u0s::Vector{U}`: Initial conditions (one per ODE, normalized)
- `tspan::Tuple{T, T}`: Time span
- `ps::Vector{P}`: Parameters (one per ODE, normalized)
- `opts::Opts`: Mode-specific options

# Constructors
```julia
# Default: Ensemble mode
LockstepProblem(lf, u0s, tspan, p=nothing)

# Explicit Ensemble mode
LockstepProblem{Ensemble}(lf, u0s, tspan, p=nothing)

# Batched mode with options
LockstepProblem{Batched}(lf, u0s, tspan, p=nothing; ordering=PerODE(), internal_threading=true)
```

# Examples
```julia
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
end

lf = LockstepFunction(lorenz!, 3, 10)
u0s = [[1.0, 0.0, 0.0] for _ in 1:10]
p = (10.0, 28.0, 8/3)

# Ensemble mode (default) - N independent integrators
prob_e = LockstepProblem(lf, u0s, (0.0, 10.0), p)
sol_e = solve(prob_e, Tsit5())

# Batched mode - single batched integrator, GPU-compatible
prob_b = LockstepProblem{Batched}(lf, u0s, (0.0, 10.0), p)
sol_b = solve(prob_b, Tsit5())
```
"""
struct LockstepProblem{M<:LockstepMode, LF, U, T, P, Opts}
    lf::LF
    u0s::Vector{U}
    tspan::Tuple{T, T}
    ps::Vector{P}
    opts::Opts
end

#==============================================================================#
# Ensemble Mode Constructor (Default)
#==============================================================================#

"""
    LockstepProblem(lf, u0s, tspan, p=nothing)

Construct a LockstepProblem with default Ensemble mode.
"""
function LockstepProblem(
    lf::LockstepFunction{F, C},
    u0s,
    tspan::Tuple,
    p = nothing
) where {F, C}
    return LockstepProblem{Ensemble}(lf, u0s, tspan, p)
end

"""
    LockstepProblem{Ensemble}(lf, u0s, tspan, p=nothing)

Construct a LockstepProblem in Ensemble mode.

In Ensemble mode, N independent ODE integrators are created and solved in parallel.
Each integrator can have different adaptive timesteps; synchronization is via
lockstep time tracking (minimum time across all integrators).
"""
function LockstepProblem{Ensemble}(
    lf::LockstepFunction{F, C},
    u0s,
    tspan::Tuple,
    p = nothing
) where {F, C}
    u0s_normalized = _normalize_u0s(u0s, lf.num_odes, lf.ode_size)
    ps_normalized = _normalize_params(p, lf.num_odes)

    U = eltype(u0s_normalized)
    T = promote_type(typeof(tspan[1]), typeof(tspan[2]))
    P = eltype(ps_normalized)

    return LockstepProblem{Ensemble, typeof(lf), U, T, P, Nothing}(
        lf,
        u0s_normalized,
        (T(tspan[1]), T(tspan[2])),
        ps_normalized,
        nothing
    )
end

#==============================================================================#
# Batched Mode Constructor
#==============================================================================#

"""
    LockstepProblem{Batched}(lf, u0s, tspan, p=nothing; ordering=PerODE(), internal_threading=true)

Construct a LockstepProblem in Batched mode.

In Batched mode, a single ODE integrator is created with a batched state vector.
RHS evaluation is parallelized across ODEs. Supports GPU acceleration.

# Keyword Arguments
- `ordering::MemoryOrdering=PerODE()`: Memory layout (PerODE or PerIndex)
- `internal_threading::Bool=true`: Enable CPU threading for RHS evaluation
"""
function LockstepProblem{Batched}(
    lf::LockstepFunction{F, C},
    u0s,
    tspan::Tuple,
    p = nothing;
    ordering::MemoryOrdering = PerODE(),
    internal_threading::Bool = true
) where {F, C}
    u0s_normalized = _normalize_u0s(u0s, lf.num_odes, lf.ode_size)
    ps_normalized = _normalize_params(p, lf.num_odes)
    opts = BatchedOpts(; ordering=ordering, internal_threading=internal_threading)

    U = eltype(u0s_normalized)
    T = promote_type(typeof(tspan[1]), typeof(tspan[2]))
    P = eltype(ps_normalized)
    O = typeof(opts)

    return LockstepProblem{Batched, typeof(lf), U, T, P, O}(
        lf,
        u0s_normalized,
        (T(tspan[1]), T(tspan[2])),
        ps_normalized,
        opts
    )
end

# ============================================================================
# Input normalization helpers (moved from solve.jl)
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
