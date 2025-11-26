module LockstepODE

using SciMLBase
using OrdinaryDiffEq
using CommonSolve

include("types.jl")
include("problem.jl")
include("integrator.jl")
include("commonsolve.jl")
include("solution.jl")

# Core types
export LockstepFunction, LockstepProblem, LockstepIntegrator, LockstepSolution

# CommonSolve interface
export init, solve, solve!, step!, reinit!

# Solution utilities
export extract_solutions, extract_at_time

end
