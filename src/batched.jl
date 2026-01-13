"""
Batched mode components for LockstepODE.

Provides BatchedFunction wrapper, SubIntegrator for callbacks, and parallel RHS evaluation.
"""

using OhMyThreads: tforeach
using OrdinaryDiffEq: DiscreteCallback, ContinuousCallback, CallbackSet

#==============================================================================#
# BatchedFunction - Callable wrapper for batched ODE solving
#==============================================================================#

"""
    BatchedFunction{O<:MemoryOrdering, F, C, P}

Callable wrapper for batched ODE solving with a LockstepFunction.

Wraps a LockstepFunction with batched-specific options (memory ordering, threading)
and provides the callable interface `(du, u, p, t)` for OrdinaryDiffEq.jl.

# Fields
- `lf::LockstepFunction{F,C}`: The underlying LockstepFunction
- `ordering::O`: Memory layout (PerODE or PerIndex)
- `internal_threading::Bool`: Enable CPU threading for RHS evaluation
- `params::P`: Per-ODE parameters (stored here to bypass SciMLBase introspection)

# Example
```julia
lf = LockstepFunction(lorenz!, 3, 100)
params = [p1, p2, ..., p100]
bf = BatchedFunction(lf, params; ordering=PerODE())

# Now bf can be used with ODEProblem (pass nothing for p)
u0_batched = vcat([[1.0, 0.0, 0.0] for _ in 1:100]...)
prob = ODEProblem(bf, u0_batched, (0.0, 10.0), nothing)
```
"""
struct BatchedFunction{O<:MemoryOrdering, F, C, P}
    lf::LockstepFunction{F, C}
    ordering::O
    internal_threading::Bool
    params::P
end

function BatchedFunction(
    lf::LockstepFunction{F, C},
    params::P;
    ordering::MemoryOrdering=PerODE(),
    internal_threading::Bool=true
) where {F, C, P}
    return BatchedFunction{typeof(ordering), F, C, P}(lf, ordering, internal_threading, params)
end

function BatchedFunction(
    lf::LockstepFunction{F, C},
    opts::BatchedOpts{O},
    params::P
) where {F, C, O, P}
    return BatchedFunction{O, F, C, P}(lf, opts.ordering, opts.internal_threading, params)
end

# Convenience accessors
num_odes(bf::BatchedFunction) = bf.lf.num_odes
ode_size(bf::BatchedFunction) = bf.lf.ode_size

#==============================================================================#
# Index Calculation - dispatched on memory ordering
#==============================================================================#

"""
    _get_idxs(bf::BatchedFunction, i::Int) -> UnitRange/StepRange

Get the indices for the i-th ODE's state in the batched state vector.

Dispatches on memory ordering:
- PerODE: Returns contiguous range `((i-1)*M+1):(i*M)`
- PerIndex: Returns strided range `i:N:((M-1)*N+i)`
"""
@inline function _get_idxs(bf::BatchedFunction{PerODE}, i::Int)
    M = bf.lf.ode_size
    return ((i - 1) * M + 1):(i * M)
end

@inline function _get_idxs(bf::BatchedFunction{PerIndex}, i::Int)
    N = bf.lf.num_odes
    M = bf.lf.ode_size
    return i:N:((M - 1) * N + i)
end

#==============================================================================#
# Callback Extraction
#==============================================================================#

# Note: _get_ode_parameters is defined in types.jl

"""
    _get_ode_callback(callbacks, i::Int, num_odes::Int)

Extract callback for the i-th ODE from callback vector.
"""
@inline function _get_ode_callback(callbacks::AbstractVector, i::Int, num_odes::Int)
    return length(callbacks) == num_odes ? callbacks[i] : callbacks
end

@inline _get_ode_callback(callbacks, ::Int, ::Int) = callbacks
@inline _get_ode_callback(::Nothing, ::Int, ::Int) = nothing

#==============================================================================#
# ODE Kernel - Core evaluation for single ODE
#==============================================================================#

"""
    ode_kernel!(i, bf::BatchedFunction, du, u, p, t)

Evaluate the i-th ODE's RHS within the batched system.

This is the core kernel called for each ODE in the batch.
Marked `@inline` for use in GPU extensions via KernelAbstractions.jl.

Note: The `p` argument is unused - parameters are stored in `bf.params`.
This signature is maintained for API compatibility with GPU extensions.
"""
@inline function ode_kernel!(i::Int, bf::BatchedFunction, du, u, p, t)
    idxs = _get_idxs(bf, i)
    u_i = view(u, idxs)
    du_i = view(du, idxs)
    p_i = bf.params[i]
    bf.lf.f(du_i, u_i, p_i, t)
    return nothing
end

#==============================================================================#
# Callable Interface
#==============================================================================#

"""
    (bf::BatchedFunction)(du, u, p, t)

Evaluate all ODEs in the batch.

Uses OhMyThreads.tforeach for parallel CPU execution when `internal_threading=true`.
For GPU arrays, dispatch is handled by extensions (CUDA, AMDGPU, Metal, oneAPI).
"""
function (bf::BatchedFunction)(du, u, p, t)
    N = bf.lf.num_odes
    if bf.internal_threading
        tforeach(i -> ode_kernel!(i, bf, du, u, p, t), 1:N)
    else
        foreach(i -> ode_kernel!(i, bf, du, u, p, t), 1:N)
    end
    return nothing
end

#==============================================================================#
# SubIntegrator - Per-ODE view for callbacks
#==============================================================================#

"""
    SubIntegrator{I, BF, U, P}

Lightweight wrapper providing a per-ODE view of the batched integrator state.

Used by the callback system to give per-ODE callbacks access to only their
ODE's state while maintaining the ability to modify the full integrator.

# Fields
- `parent::I`: The full OrdinaryDiffEq integrator
- `bf::BF`: The BatchedFunction containing metadata
- `ode_idx::Int`: Index of the ODE this sub-integrator represents
- `u::U`: View of the state for this ODE only
- `p::P`: Parameters for this ODE
"""
mutable struct SubIntegrator{I, BF, U, P}
    parent::I
    bf::BF
    ode_idx::Int
    u::U
    p::P
end

function SubIntegrator(integrator, bf::BatchedFunction, ode_idx::Int)
    idxs = _get_idxs(bf, ode_idx)
    u_view = view(integrator.u, idxs)
    p_i = bf.params[ode_idx]
    return SubIntegrator(integrator, bf, ode_idx, u_view, p_i)
end

# Forward property access to parent integrator
function Base.getproperty(sub::SubIntegrator, sym::Symbol)
    if sym in (:parent, :bf, :ode_idx, :u, :p)
        return getfield(sub, sym)
    elseif sym === :t
        return getfield(sub, :parent).t
    else
        return getproperty(getfield(sub, :parent), sym)
    end
end

# Allow setting properties on parent integrator
function Base.setproperty!(sub::SubIntegrator, sym::Symbol, val)
    if sym in (:parent, :bf, :ode_idx, :u, :p)
        return setfield!(sub, sym, val)
    else
        return setproperty!(getfield(sub, :parent), sym, val)
    end
end

#==============================================================================#
# Callback Wrapping
#==============================================================================#

"""
    create_lockstep_callbacks(bf::BatchedFunction)

Create wrapped callbacks for batched ODE solving.

Takes a BatchedFunction with per-ODE callbacks stored in its LockstepFunction
and returns callbacks that dispatch to SubIntegrators for each ODE.

Returns `nothing` if no callbacks are stored.
"""
function create_lockstep_callbacks(bf::BatchedFunction)
    lf = bf.lf
    isnothing(lf.callbacks) && return nothing

    wrapped_callbacks = []

    for i in 1:lf.num_odes
        cb = _get_ode_callback(lf.callbacks, i, lf.num_odes)
        isnothing(cb) && continue

        if cb isa DiscreteCallback || cb isa ContinuousCallback
            push!(wrapped_callbacks, _wrap_callback(cb, bf, i))
        else
            error("Unsupported callback type: $(typeof(cb)). " *
                  "Only DiscreteCallback and ContinuousCallback are supported.")
        end
    end

    if length(wrapped_callbacks) == 0
        return nothing
    elseif length(wrapped_callbacks) == 1
        return wrapped_callbacks[1]
    else
        return CallbackSet(wrapped_callbacks...)
    end
end

# Shared callback wrapping helpers
@inline function _wrap_condition(bf::BatchedFunction, ode_idx::Int, condition)
    return function (_u, _t, integrator)
        sub = SubIntegrator(integrator, bf, ode_idx)
        return condition(sub.u, sub.t, sub)
    end
end

@inline function _wrap_affect(bf::BatchedFunction, ode_idx::Int, affect!)
    return function (integrator)
        sub = SubIntegrator(integrator, bf, ode_idx)
        affect!(sub)
        return nothing
    end
end

function _wrap_callback(cb::DiscreteCallback, bf::BatchedFunction, ode_idx::Int)
    return DiscreteCallback(
        _wrap_condition(bf, ode_idx, cb.condition),
        _wrap_affect(bf, ode_idx, cb.affect!);
        save_positions=cb.save_positions
    )
end

function _wrap_callback(cb::ContinuousCallback, bf::BatchedFunction, ode_idx::Int)
    wrapped_affect_neg! = isnothing(cb.affect_neg!) ? nothing :
                          _wrap_affect(bf, ode_idx, cb.affect_neg!)
    return ContinuousCallback(
        _wrap_condition(bf, ode_idx, cb.condition),
        _wrap_affect(bf, ode_idx, cb.affect!),
        wrapped_affect_neg!;
        save_positions=cb.save_positions
    )
end

#==============================================================================#
# Batched Initial Conditions
#==============================================================================#

"""
    batch_u0s(u0s::Vector{<:AbstractVector}, ::PerODE)

Concatenate per-ODE initial conditions into PerODE layout.
Layout: [u1_ode1, u2_ode1, ..., u1_ode2, u2_ode2, ...]
"""
function batch_u0s(u0s::Vector{<:AbstractVector}, ::PerODE)
    return vcat(u0s...)
end

"""
    batch_u0s(u0s::Vector{<:AbstractVector}, ::PerIndex)

Concatenate per-ODE initial conditions into PerIndex layout.
Layout: [u1_ode1, u1_ode2, ..., u2_ode1, u2_ode2, ...]
"""
function batch_u0s(u0s::Vector{<:AbstractVector}, ::PerIndex)
    N = length(u0s)
    M = length(first(u0s))
    T = eltype(first(u0s))
    result = Vector{T}(undef, N * M)
    for j in 1:M
        for i in 1:N
            result[(j - 1) * N + i] = u0s[i][j]
        end
    end
    return result
end

# Default to PerODE for backward compatibility
function batch_u0s(u0s::Vector{<:AbstractVector})
    return batch_u0s(u0s, PerODE())
end
