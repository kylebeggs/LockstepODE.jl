module LockstepODECUDAExt

using LockstepODE
import KernelAbstractions as KA
using CUDA

# GPU kernel for parallel ODE execution
KA.@kernel function ode_kernel!(lockstep_func, du, u, p, t)
    i = KA.@index(Global)
    LockstepODE.ode_kernel!(i, lockstep_func, du, u, p, t)
end

# CUDA-specific dispatch
function (lockstep_func::LockstepODE.LockstepFunction{O, F, CUDABackend})(
        du, u, p, t) where {O, F}
    backend = KA.get_backend(u)
    N = lockstep_func.num_odes

    # Launch GPU kernel with one thread per ODE
    kernel = ode_kernel!(backend)
    kernel(lockstep_func, du, u, p, t, ndrange = N)
    KA.synchronize(backend)
    return nothing
end

end
