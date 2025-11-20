"""
    AbstractODEWrapper

Abstract type for wrapping different kinds of ODE functions.

Concrete subtypes must implement:
- Constructor from the function/system
- Method to call the wrapped function with signature `(wrapper, du, u, p, t)`
"""
abstract type AbstractODEWrapper end

"""
    SimpleWrapper{F}

Wrapper for regular Julia functions with signature `f(du, u, p, t)`.

This is the default wrapper for plain Julia functions and maintains
backward compatibility with the original LockstepODE design.
"""
struct SimpleWrapper{F} <: AbstractODEWrapper
    f::F
end

"""
    (wrapper::SimpleWrapper)(du, u, p, t)

Call the wrapped function directly with the provided arguments.
Works with array views for efficient parallel execution.
"""
@inline function (wrapper::SimpleWrapper)(du, u, p, t)
    wrapper.f(du, u, p, t)
    return nothing
end

"""
    param_size(wrapper::AbstractODEWrapper)

Get the number of parameters for the ODE system.

For `SimpleWrapper`, returns 0 since plain Julia functions don't expose parameter metadata.
For `MTKWrapper`, returns the actual parameter count from the ModelingToolkit system.
"""
param_size(::SimpleWrapper) = 0
