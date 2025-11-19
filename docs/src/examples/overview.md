# Examples Overview

This section provides comprehensive, worked examples demonstrating the key features of LockstepODE.jl. Each example includes mathematical problem formulation, detailed implementation, and expected results.

## Available Examples

### [Per-ODE Callbacks](@ref)
Learn how to apply different callbacks to individual ODEs in your batched system. This example demonstrates:
- Assigning different reset thresholds to each ODE
- Using shared callbacks across all ODEs
- Combining per-ODE callbacks with different parameters
- Thread-safe callback design patterns

**Recommended for**: Users who need event handling with different conditions per ODE

### [Basic Usage](@ref)
Master the fundamental patterns for working with LockstepODE. This example covers:
- Setting up multiple initial conditions
- Batching parameters for different ODEs
- Using utility functions for efficient data preparation
- Extracting and analyzing individual solutions

**Recommended for**: New users getting started with LockstepODE

### [Advanced Configuration](@ref)
Optimize performance and control parallel execution. This example explores:
- Memory layout options (`PerODE` vs `PerIndex`)
- Threading control and external parallelization
- Performance considerations for different use cases
- Cache-friendly data organization

**Recommended for**: Users optimizing performance or integrating with existing parallel code

## Getting Help

If you encounter issues or have questions:
- Check the [API Reference](@ref) for function signatures and options
- Review the [Getting Started](@ref) guide for workflow overview
- Open an issue on [GitHub](https://github.com/kylebeggs/LockstepODE.jl/issues)
