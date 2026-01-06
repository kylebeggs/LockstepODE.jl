module LockstepODEMetalExt

using LockstepODE
using Metal
import KernelAbstractions as KA

"""
GPU kernel for batched ODE evaluation on Apple Metal GPUs.

Wraps the core `ode_kernel!` function for execution on Apple GPUs.
"""
KA.@kernel function lockstep_kernel_gpu!(bf, du, u, p, t)
    i = KA.@index(Global)
    LockstepODE.ode_kernel!(i, bf, du, u, p, t)
end

"""
Dispatch for MtlArray - enables automatic GPU execution for Apple GPUs.

When `u` and `du` are `MtlArray`s, the batched ODE evaluation is performed
on the GPU using KernelAbstractions.jl.
"""
function (bf::LockstepODE.BatchedFunction)(du::MtlArray, u::MtlArray, p, t)
    backend = KA.get_backend(u)
    N = bf.lf.num_odes

    kernel = lockstep_kernel_gpu!(backend)
    kernel(bf, du, u, p, t; ndrange=N)
    KA.synchronize(backend)

    return nothing
end

end # module
