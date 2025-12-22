module LockstepODE

using SciMLBase
using OrdinaryDiffEq
using CommonSolve

# Include order matters: types → batched (uses types) → problem (uses BatchedOpts) → ...
include("types.jl")
include("batched.jl")
include("problem.jl")
include("integrator.jl")
include("commonsolve.jl")
include("solution.jl")

#==============================================================================#
# Exports
#==============================================================================#

# Mode types
export LockstepMode, Ensemble, Batched

# Memory ordering types (for Batched mode)
export MemoryOrdering, PerODE, PerIndex

# Options types
export BatchedOpts

# Core types
export LockstepFunction, LockstepProblem, LockstepSolution

# Integrator types
export LockstepIntegrator, AbstractLockstepIntegrator
export EnsembleLockstepIntegrator, BatchedLockstepIntegrator

# Batched mode internals (for extensions)
export BatchedFunction, SubIntegrator, BatchedSubSolution
export ode_kernel!, _get_idxs

# CommonSolve interface
export init, solve, solve!, step!, reinit!

# Solution utilities
export extract_solutions, extract_at_time

end
