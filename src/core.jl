using OrdinaryDiffEq
using OhMyThreads
import KernelAbstractions as KA

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
    LockstepFunction{O, F, B}

A wrapper that enables parallel execution of multiple instances of the same ODE system.

This struct implements the callable interface required by OrdinaryDiffEq.jl, allowing 
you to solve multiple ODEs in parallel by batching them into a single larger system.

# Type Parameters
- `O`: Memory ordering type (`PerODE` or `PerIndex`)
- `F`: Type of the wrapped ODE function
- `B`: Backend type for computation (e.g., `KA.CPU()`)

# Fields
- `f::F`: The ODE function to be applied to each system
- `num_odes::Int`: Number of ODE systems to solve in parallel
- `ode_size::Int`: Size of each individual ODE system
- `internal_threading::Bool`: Whether to use internal threading for parallel execution
- `ordering::O`: Memory layout ordering (`PerODE` or `PerIndex`)
- `backend::B`: Computational backend (default: `KA.CPU()`)

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
struct LockstepFunction{O, F, B}
    f::F
    num_odes::Int
    ode_size::Int
    internal_threading::Bool
    ordering::O
    backend::B
end

"""
    LockstepFunction(f, ode_size::Int, num_odes::Int; 
                     internal_threading=true, ordering=nothing, backend=KA.CPU())

Construct a `LockstepFunction` for parallel ODE solving.

# Arguments
- `f`: The ODE function with signature `f(du, u, p, t)`
- `ode_size::Int`: The number of state variables in each ODE system
- `num_odes::Int`: The number of ODE systems to solve in parallel

# Keyword Arguments
- `internal_threading::Bool = true`: Enable internal threading for parallel execution
- `ordering = nothing`: Memory ordering (`PerODE` or `PerIndex`). If `nothing`, defaults to `PerODE()`
- `backend = KA.CPU()`: Computational backend for kernel execution

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
"""
function LockstepFunction(
        f,
        ode_size::Int,
        num_odes::Int;
        internal_threading = true,
        ordering = nothing,
        backend = KA.CPU()
)
    _ordering_actual = ordering === nothing ? _ordering(backend) : ordering

    return LockstepFunction(
        f, num_odes, ode_size, internal_threading, _ordering_actual, backend)
end

_ordering(::KA.CPU) = PerODE()

function ode_kernel!(i, lockstep_func::LockstepFunction, du, u, p, t)
    idxs = _get_idxs(lockstep_func, i)
    u_i = view(u, idxs)
    du_i = view(du, idxs)
    p_i = _get_ode_parameters(p, i, lockstep_func.num_odes)
    lockstep_func.f(du_i, u_i, p_i, t)
    return nothing
end

function (lockstep_func::LockstepFunction{O, F, KA.CPU})(du, u, p, t) where {O, F}
    N = lockstep_func.num_odes
    if lockstep_func.internal_threading
        OhMyThreads.tforeach(i -> ode_kernel!(i, lockstep_func, du, u, p, t), 1:N)
    else
        foreach(i -> ode_kernel!(i, lockstep_func, du, u, p, t), 1:N)
    end
    return nothing
end
