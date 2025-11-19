module LockstepODEAMDGPUExt

using LockstepODE
import KernelAbstractions as KA
using AMDGPU


KA.@kernel function ode_kernel!(lockstep_func, du, u, p, t)
    i = KA.@index(Global)
    LockstepODE.ode_kernel!(i, lockstep_func, du, u, p, t)
end

# AMDGPU/ROCm array dispatch - automatically use GPU implementation for ROCArrays
function (lockstep_func::LockstepODE.LockstepFunction{O, F})(
        du::ROCArray, u::ROCArray, p, t) where {O, F}
    backend = KA.get_backend(u)
    N = lockstep_func.num_odes

    # Calculate optimal workgroup size for GPU
    workgroup_size = min(256, N)  # Common GPU warp/wavefront size

    kernel = ode_kernel!(backend)
    kernel(lockstep_func, du, u, p, t; ndrange = N, workgroupsize = workgroup_size)
    KA.synchronize(backend)
    return nothing
end

end
