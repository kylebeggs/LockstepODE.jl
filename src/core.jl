using OrdinaryDiffEq
using OhMyThreads
using OrdinaryDiffEq: DiscreteCallback, ContinuousCallback, CallbackSet
import OrdinaryDiffEq: ODEProblem

# Core types and structures

"""
    PerODE

Memory ordering where variables for each ODE are stored contiguously.

This is the default and generally recommended ordering. With this layout, all state 
variables for a single ODE system are stored together in memory, which typically 
provides better performance for most ODE functions.

# Example
For 2 ODEs with 3 variables each, the memory layout is:
`[u1_ode1, u2_ode1, u3_ode1, u1_ode2, u2_ode2, u3_ode2]`
"""
struct PerODE end

"""
    PerIndex

Memory ordering where variables of the same index across all ODEs are stored contiguously.

This ordering can provide better cache locality for certain operations that access the 
same variable across multiple ODEs. Use this when your ODE function benefits from 
accessing the same state variable across different ODE instances.

# Example
For 2 ODEs with 3 variables each, the memory layout is:
`[u1_ode1, u1_ode2, u2_ode1, u2_ode2, u3_ode1, u3_ode2]`
"""
struct PerIndex end

"""
    LockstepFunction{O, W<:AbstractODEWrapper, C}

A wrapper that enables parallel execution of multiple instances of the same ODE system.

This struct implements the callable interface required by OrdinaryDiffEq.jl, allowing
you to solve multiple ODEs in parallel by batching them into a single larger system.

# Type Parameters
- `O`: Memory ordering type (`PerODE` or `PerIndex`)
- `W`: Type of the ODE wrapper (e.g., `SimpleWrapper` for regular functions, `MTKWrapper` for ModelingToolkit)
- `C`: Type of the callbacks (Nothing or AbstractVector)

# Fields
- `wrapper::W`: The wrapper containing the ODE function/system
- `num_odes::Int`: Number of ODE systems to solve in parallel
- `ode_size::Int`: Size of each individual ODE system
- `internal_threading::Bool`: Whether to use internal threading for parallel execution
- `ordering::O`: Memory layout ordering (`PerODE` or `PerIndex`)
- `callbacks::C`: Optional callbacks to apply to each ODE (per-ODE or shared)

# Example
```julia
# Define a simple ODE function
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Create a LockstepFunction for 100 parallel Lorenz systems
lockstep_func = LockstepFunction(lorenz!, 3, 100)
```
"""
struct LockstepFunction{O, W<:AbstractODEWrapper, C}
    wrapper::W
    num_odes::Int
    ode_size::Int
    internal_threading::Bool
    ordering::O
    callbacks::C
end

"""
    LockstepFunction(f, ode_size::Int, num_odes::Int;
                     internal_threading=true, ordering=PerODE(), callbacks=nothing)

Construct a `LockstepFunction` for parallel ODE solving.

# Arguments
- `f`: The ODE function with signature `f(du, u, p, t)`
- `ode_size::Int`: The number of state variables in each ODE system
- `num_odes::Int`: The number of ODE systems to solve in parallel

# Keyword Arguments
- `internal_threading::Bool = true`: Enable internal threading for parallel execution
- `ordering = PerODE()`: Memory ordering (`PerODE` or `PerIndex`)
- `callbacks = nothing`: Optional callbacks to apply to ODEs. Can be:
  - `nothing`: No callbacks
  - Single callback: Applied to all ODEs
  - Vector of callbacks: Each callback applied to corresponding ODE (length must equal `num_odes`)

# Returns
A `LockstepFunction` instance that can be passed to `ODEProblem`

# Example
```julia
# Define ODE function
function simple_decay!(du, u, p, t)
    du[1] = -p * u[1]
end

# Create lockstep function for 50 decay ODEs, each with 1 variable
lockstep_func = LockstepFunction(simple_decay!, 1, 50)

# Use with batched initial conditions
u0_batched = ones(50)  # 50 ODEs, each starting at 1.0
tspan = (0.0, 10.0)
p = 0.1  # Same decay rate for all

prob = ODEProblem(lockstep_func, u0_batched, tspan, p)
sol = solve(prob, Tsit5())
```

# Example with callbacks
```julia
using OrdinaryDiffEq

# Define a callback that resets state when it exceeds 10
condition(u, t, integrator) = u[1] > 10.0
affect!(integrator) = (integrator.u[1] = 0.0)
reset_cb = DiscreteCallback(condition, affect!)

# Apply the same callback to all ODEs
lockstep_func = LockstepFunction(simple_decay!, 1, 50, callbacks=reset_cb)

# Or use different callbacks for each ODE
callbacks_vec = [DiscreteCallback(...) for _ in 1:50]
lockstep_func = LockstepFunction(simple_decay!, 1, 50, callbacks=callbacks_vec)
```

# Thread Safety
When `internal_threading=true`, ODEs are evaluated in parallel. Per-ODE callbacks should be
thread-safe, avoiding modifications to shared state. Callbacks that only modify their own
ODE's state (via `integrator.u`) are safe by design.
"""
function LockstepFunction(
        f,
        ode_size::Int,
        num_odes::Int;
        internal_threading = true,
        ordering = PerODE(),
        callbacks = nothing
)
    # Wrap the function in a SimpleWrapper (default for regular Julia functions)
    wrapper = SimpleWrapper(f)
    return LockstepFunction(wrapper, num_odes, ode_size, internal_threading, ordering, callbacks)
end

"""
    SubIntegrator

A lightweight wrapper that presents a view of a single ODE's state from a batched integrator.

This struct is used internally by the callback system to provide per-ODE callbacks with
access to only their ODE's state, while maintaining the ability to modify the full
integrator's state.

Implements SymbolicIndexingInterface for compatibility with ModelingToolkit's `getu()` and
other symbolic utilities (when used with MTK systems).

# Fields
- `parent`: The full integrator from OrdinaryDiffEq.jl
- `lockstep_func`: The LockstepFunction containing metadata
- `ode_idx::Int`: Index of the ODE this sub-integrator represents
- `u`: View of the state for this ODE only
- `p`: Parameters for this ODE (extracted using _get_ode_parameters)
"""
mutable struct SubIntegrator{I, L, U, P}
    parent::I
    lockstep_func::L
    ode_idx::Int
    u::U
    p::P
end

function SubIntegrator(integrator, lockstep_func, ode_idx)
    idxs = _get_idxs(lockstep_func, ode_idx)
    u_view = view(integrator.u, idxs)
    p_i = _get_ode_parameters(integrator.p, ode_idx, lockstep_func.num_odes, param_size(lockstep_func.wrapper))
    return SubIntegrator(integrator, lockstep_func, ode_idx, u_view, p_i)
end

# Forward property access to parent integrator for common fields
Base.getproperty(sub::SubIntegrator, sym::Symbol) = begin
    if sym in (:parent, :lockstep_func, :ode_idx, :u, :p)
        getfield(sub, sym)
    elseif sym === :t
        getfield(sub, :parent).t
    else
        getfield(getfield(sub, :parent), sym)
    end
end

"""
    create_lockstep_callbacks(lockstep_func::LockstepFunction)

Create callbacks that wrap per-ODE callbacks for use with batched ODE solving.

This function takes a `LockstepFunction` with per-ODE callbacks and returns a callback
(or CallbackSet) that can be passed to `solve()`. Each per-ODE callback receives a
`SubIntegrator` that only exposes its own ODE's state.

# Arguments
- `lockstep_func::LockstepFunction`: LockstepFunction containing callbacks

# Returns
A callback compatible with OrdinaryDiffEq.jl that dispatches to per-ODE callbacks,
or `nothing` if no callbacks are stored.

# Example
```julia
# Define per-ODE callbacks
cb1 = DiscreteCallback((u, t, integrator) -> u[1] > 10, integrator -> (integrator.u[1] = 0))
cb2 = DiscreteCallback((u, t, integrator) -> u[1] < -5, integrator -> (integrator.u[1] = 0))

# Create lockstep function with callbacks
lockstep_func = LockstepFunction(f, ode_size, num_odes, callbacks=[cb1, cb2, ...])

# Create wrapped callbacks for solver
wrapped_cb = create_lockstep_callbacks(lockstep_func)

# Use with solve
prob = ODEProblem(lockstep_func, u0, tspan, p)
sol = solve(prob, Tsit5(), callback=wrapped_cb)
```
"""
function create_lockstep_callbacks(lockstep_func::LockstepFunction)
    isnothing(lockstep_func.callbacks) && return nothing

    wrapped_callbacks = []

    # Pre-allocate cache for SubIntegrators to avoid allocation on every callback invocation
    # SubIntegrator.u is a view that automatically reflects integrator.u changes
    # SubIntegrator property access forwards to parent, so no updates needed after initialization
    sub_cache = Vector{Union{Nothing, SubIntegrator}}(nothing, lockstep_func.num_odes)

    for i in 1:lockstep_func.num_odes
        cb = _get_ode_callback(lockstep_func.callbacks, i, lockstep_func.num_odes)
        isnothing(cb) && continue

        # Wrap the callback for this ODE
        if cb isa DiscreteCallback
            wrapped_condition = function (u, t, integrator)
                if isnothing(sub_cache[i])
                    sub_cache[i] = SubIntegrator(integrator, lockstep_func, i)
                end
                return cb.condition(sub_cache[i].u, t, sub_cache[i])
            end

            wrapped_affect! = function (integrator)
                # SubIntegrator already initialized by condition check
                cb.affect!(sub_cache[i])
                return nothing
            end

            push!(wrapped_callbacks, DiscreteCallback(wrapped_condition, wrapped_affect!;
                                                     save_positions=cb.save_positions))
        elseif cb isa ContinuousCallback
            wrapped_condition = function (u, t, integrator)
                if isnothing(sub_cache[i])
                    sub_cache[i] = SubIntegrator(integrator, lockstep_func, i)
                end
                return cb.condition(sub_cache[i].u, t, sub_cache[i])
            end

            wrapped_affect! = function (integrator)
                # SubIntegrator already initialized by condition check
                cb.affect!(sub_cache[i])
                return nothing
            end

            if !isnothing(cb.affect_neg!)
                wrapped_affect_neg! = function (integrator)
                    # SubIntegrator already initialized by condition check
                    cb.affect_neg!(sub_cache[i])
                    return nothing
                end
            else
                wrapped_affect_neg! = nothing
            end

            push!(wrapped_callbacks, ContinuousCallback(wrapped_condition, wrapped_affect!,
                                                       wrapped_affect_neg!;
                                                       save_positions=cb.save_positions))
        else
            error("Unsupported callback type: $(typeof(cb))")
        end
    end

    # Return single callback or CallbackSet
    if length(wrapped_callbacks) == 1
        return wrapped_callbacks[1]
    elseif length(wrapped_callbacks) > 1
        return CallbackSet(wrapped_callbacks...)
    else
        return nothing
    end
end

"""
    ODEProblem(f::LockstepFunction, u0, tspan, [p]; callback=nothing, kwargs...)

Construct an ODEProblem with a LockstepFunction, automatically handling per-ODE callbacks.

This constructor automatically wraps callbacks stored in the LockstepFunction and merges them
with any user-provided callbacks. The wrapped callbacks ensure each ODE receives its own
SubIntegrator with isolated state access.

# Arguments
- `f::LockstepFunction`: The lockstep function containing ODE logic and optional callbacks
- `u0`: Initial conditions (batched format expected)
- `tspan`: Time span tuple (tstart, tend)
- `p`: Parameters (optional, can be shared or per-ODE)

# Keyword Arguments
- `callback=nothing`: Optional additional callbacks to merge with LockstepFunction callbacks
- `kwargs...`: Other ODEProblem keyword arguments

# Example
```julia
# Callbacks are automatically handled - no need for create_lockstep_callbacks()
function growth!(du, u, p, t)
    du[1] = p * u[1]
end

reset_cb = DiscreteCallback(
    (u, t, integrator) -> u[1] > 10.0,
    integrator -> (integrator.u[1] = 1.0)
)

lockstep_func = LockstepFunction(growth!, 1, 5, callbacks=reset_cb)
u0 = ones(5)
prob = ODEProblem(lockstep_func, u0, (0.0, 10.0), 1.0)
sol = solve(prob, Tsit5())  # Callbacks applied automatically!
```
"""
function ODEProblem(f::LockstepFunction, u0, tspan, p; callback=nothing, kwargs...)
    # Extract and wrap callbacks from LockstepFunction
    lockstep_cb = create_lockstep_callbacks(f)

    # Merge with user-provided callback
    merged_callback = if !isnothing(lockstep_cb) && !isnothing(callback)
        CallbackSet(lockstep_cb, callback)
    elseif !isnothing(lockstep_cb)
        lockstep_cb
    else
        callback
    end

    # Call base ODEProblem constructor by invoking the generic method
    # This avoids infinite recursion by explicitly calling the parent method
    return invoke(ODEProblem, Tuple{Any, Any, Any, Any}, f, u0, tspan, p; callback=merged_callback, kwargs...)
end

# Method without parameters (default to nothing, which ODEProblem handles)
function ODEProblem(f::LockstepFunction, u0, tspan; callback=nothing, kwargs...)
    return ODEProblem(f, u0, tspan, nothing; callback=callback, kwargs...)
end


@inline function ode_kernel!(i, lockstep_func::LockstepFunction, du, u, p, t)
    idxs = _get_idxs(lockstep_func, i)
    u_i = view(u, idxs)
    du_i = view(du, idxs)
    p_i = _get_ode_parameters(p, i, lockstep_func.num_odes, param_size(lockstep_func.wrapper))
    # Pass ODE index to wrapper for per-ODE parameter support
    # Wrappers can dispatch on the 5-arg vs 4-arg signature
    _call_wrapper(lockstep_func.wrapper, du_i, u_i, p_i, t, i)
    return nothing
end

# Default wrapper call (4-arg signature for backwards compatibility)
@inline _call_wrapper(wrapper, du, u, p, t, i) = wrapper(du, u, p, t)

function (lockstep_func::LockstepFunction{O, W, C})(du, u, p, t) where {O, W, C}
    N = lockstep_func.num_odes
    if lockstep_func.internal_threading
        OhMyThreads.tforeach(i -> ode_kernel!(i, lockstep_func, du, u, p, t), 1:N)
    else
        foreach(i -> ode_kernel!(i, lockstep_func, du, u, p, t), 1:N)
    end
    return nothing
end
