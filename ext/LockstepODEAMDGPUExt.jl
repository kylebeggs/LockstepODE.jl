module LockstepODEAMDGPUExt

using LockstepODE
using AMDGPU
import KernelAbstractions as KA

"""
GPU kernel for batched ODE evaluation on AMD GPUs.

Wraps the core `ode_kernel!` function for execution on AMD GPUs.
"""
KA.@kernel function lockstep_kernel_gpu!(bf, du, u, p, t)
    i = KA.@index(Global)
    LockstepODE.ode_kernel!(i, bf, du, u, p, t)
end

"""
Dispatch for ROCArray - enables automatic GPU execution for AMD GPUs.

When `u` and `du` are `ROCArray`s, the batched ODE evaluation is performed
on the GPU using KernelAbstractions.jl.
"""
function (bf::LockstepODE.BatchedFunction)(du::ROCArray, u::ROCArray, p, t)
    backend = KA.get_backend(u)
    N = bf.lf.num_odes

    kernel = lockstep_kernel_gpu!(backend)
    kernel(bf, du, u, p, t; ndrange=N)
    KA.synchronize(backend)

    return nothing
end

end # module
