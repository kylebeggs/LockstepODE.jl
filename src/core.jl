using OrdinaryDiffEq
using OhMyThreads
import OrdinaryDiffEq: solve

# Core types and structures
struct PerODE end
struct PerIndex end

# Lockstep function callable struct
struct LockstepFunction{O,F}
    f::F
    num_odes::Int
    ode_size::Int
    internal_threading::Bool
    ordering::O
end

# Convenient constructor that calculates ode_size from u0
function LockstepFunction(
    f,
    u0::AbstractVector,
    num_odes::Int;
    internal_threading=true,
    ordering=PerODE(),
)
    ode_size = length(u0) ÷ num_odes
    return LockstepFunction(f, num_odes, ode_size, internal_threading, ordering)
end


# Core execution functions
function cell_kernel!(i, lockstep_func::LockstepFunction, du, u, p, t)
    idxs = _get_idxs(lockstep_func, i)
    u_i = view(u, idxs)
    du_i = view(du, idxs)
    p_i = _get_ode_parameters(p, i, lockstep_func.num_odes)
    lockstep_func.f(du_i, u_i, p_i, t)
    return nothing
end

function (lockstep_func::LockstepFunction)(du, u, p, t)
    N = lockstep_func.num_odes
    if lockstep_func.internal_threading
        OhMyThreads.tforeach(i -> cell_kernel!(i, lockstep_func, du, u, p, t), 1:N)
    else
        foreach(i -> cell_kernel!(i, lockstep_func, du, u, p, t), 1:N)
    end
    return nothing
end

