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
- `US<:AbstractVector`: Storage type for `u0s` (vector-of-vectors or pre-flattened batch)
- `T`: Time type
- `P`: Parameter element type
- `Opts`: Mode-specific options type (Nothing for Ensemble, BatchedOpts for Batched)

# Fields
- `lf::LF`: LockstepFunction coordinator
- `u0s::US`: Initial conditions. Either a `Vector{<:AbstractVector}` of per-ODE states,
  or (Batched mode only) a flat `AbstractVector{<:Number}` of length `num_odes * ode_size`
  already arranged in the configured `MemoryOrdering`.
- `tspan::Tuple{T, T}`: Time span
- `ps::Vector{P}`: Parameters (one per ODE, normalized)
- `opts::Opts`: Mode-specific options

# Constructors
```julia
# Default: Batched mode (best performance)
LockstepProblem(lf, u0s, tspan, p=nothing)

# Explicit Batched mode with options
LockstepProblem{Batched}(lf, u0s, tspan, p=nothing; ordering=PerODE(), internal_threading=true)

# Ensemble mode (independent timesteps per ODE)
LockstepProblem{Ensemble}(lf, u0s, tspan, p=nothing)
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

# Batched mode (default) - single batched integrator, GPU-compatible
prob_b = LockstepProblem(lf, u0s, (0.0, 10.0), p)
sol_b = solve(prob_b, Tsit5())

# Ensemble mode - N independent integrators with adaptive timesteps
prob_e = LockstepProblem{Ensemble}(lf, u0s, (0.0, 10.0), p)
sol_e = solve(prob_e, Tsit5())
```
"""
struct LockstepProblem{M<:LockstepMode, LF, US<:AbstractVector, T, P, Opts}
    lf::LF
    u0s::US
    tspan::Tuple{T, T}
    ps::Vector{P}
    opts::Opts
end

#==============================================================================#
# Default Constructor (Batched Mode)
#==============================================================================#

"""
    LockstepProblem(lf, u0s, tspan, p=nothing)

Construct a LockstepProblem with default Batched mode.
"""
function LockstepProblem(
    lf::LockstepFunction{F, C},
    u0s,
    tspan::Tuple,
    p = nothing
) where {F, C}
    return LockstepProblem{Batched}(lf, u0s, tspan, p)
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
    u0s isa AbstractVector{<:Number} && length(u0s) == lf.num_odes * lf.ode_size &&
        throw(ArgumentError(
            "Ensemble mode requires per-ODE initial conditions (a vector of vectors) " *
            "or a single state vector to replicate. A pre-flattened batch of length " *
            "num_odes * ode_size is only supported in Batched mode."
        ))

    u0s_normalized = _normalize_u0s(u0s, lf.num_odes, lf.ode_size)
    ps_normalized = _normalize_params(p, lf.num_odes)

    US = typeof(u0s_normalized)
    T = promote_type(typeof(tspan[1]), typeof(tspan[2]))
    P = eltype(ps_normalized)

    return LockstepProblem{Ensemble, typeof(lf), US, T, P, Nothing}(
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

    US = typeof(u0s_normalized)
    T = promote_type(typeof(tspan[1]), typeof(tspan[2]))
    P = eltype(ps_normalized)
    O = typeof(opts)

    return LockstepProblem{Batched, typeof(lf), US, T, P, O}(
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
    n = length(u0)
    if n == ode_size
        # Single u0 - replicate for all ODEs
        return [copy(u0) for _ in 1:num_odes]
    elseif n == num_odes * ode_size
        # Pre-flattened batch (Batched mode) - use as-is.
        # Caller is responsible for arranging bytes consistent with the chosen MemoryOrdering.
        return u0
    else
        throw(ArgumentError(
            "u0 must have length $ode_size (single state, replicated across ODEs) or " *
            "$(num_odes * ode_size) (pre-flattened batch of $num_odes ODEs × $ode_size state), " *
            "got $n"
        ))
    end
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
