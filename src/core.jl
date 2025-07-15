using OrdinaryDiffEq
using OhMyThreads
import KernelAbstractions as KA

# Core types and structures
struct PerODE end
struct PerIndex end

struct LockstepFunction{O, F, B}
    f::F
    num_odes::Int
    ode_size::Int
    internal_threading::Bool
    ordering::O
    backend::B
end

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
