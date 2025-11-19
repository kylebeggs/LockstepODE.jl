# Advanced Configuration

This page covers performance optimization and advanced configuration options in LockstepODE.jl, including memory layout selection and threading control.

## Memory Layouts

LockstepODE supports two memory layouts for organizing batched ODE data: `PerODE` and `PerIndex`. The choice of layout can affect cache performance and access patterns.

### Memory Layout Comparison

For a system with 3 ODEs, each with 2 variables ($u_1$, $u_2$), the layouts differ:

**PerODE (default)**: Variables for each ODE stored contiguously
```
[ODE1_u1, ODE1_u2, ODE2_u1, ODE2_u2, ODE3_u1, ODE3_u2]
```

**PerIndex**: Same variable index across ODEs stored contiguously
```
[ODE1_u1, ODE2_u1, ODE3_u1, ODE1_u2, ODE2_u2, ODE3_u2]
```

### When to Use Each Layout

**PerODE (recommended for most cases)**:
- Better cache locality when computing derivatives for a single ODE
- More intuitive data organization
- Default behavior

**PerIndex**:
- Better cache locality for operations across all ODEs on the same variable
- Useful when vectorizing operations across ODEs
- Potentially better SIMD opportunities

### Using PerIndex Layout

```julia
using LockstepODE
using OrdinaryDiffEq

function my_ode!(du, u, p, t)
    du[1] = p * u[2]
    du[2] = -u[1]
end

num_odes = 4
u0_vec = [[1.0, 0.0], [2.0, 1.0], [0.5, -0.5], [1.5, 0.8]]
u0_batched = vcat(u0_vec...)
p_batched = [1.0, 1.2, 0.8, 1.1]

# Use PerIndex layout
lockstep_func_per_index = LockstepFunction(
    my_ode!,
    2,  # 2 variables per ODE
    num_odes;
    ordering = PerIndex()
)

prob_per_index = ODEProblem(lockstep_func_per_index, u0_batched, (0.0, 10.0), p_batched)
sol_per_index = solve(prob_per_index, Tsit5())

# Extract solutions works the same way
individual_sols = extract_solutions(lockstep_func_per_index, sol_per_index)
```

### Discussion

The memory layout is primarily an implementation detail that affects performance but not correctness. Key points:

1. **Transparent to users**: The same ODE function works with both layouts
2. **Dispatch-based**: LockstepODE uses Julia's type system to dispatch to the correct indexing
3. **Benchmark both**: For performance-critical applications, test both layouts
4. **System-dependent**: Optimal layout depends on problem size, hardware, and ODE complexity

### Performance Considerations

```julia
using BenchmarkTools

function benchmark_layouts(ode_func, ode_size, num_odes, u0, p, tspan)
    # PerODE layout
    lockstep_per_ode = LockstepFunction(ode_func, ode_size, num_odes, ordering=PerODE())
    prob_per_ode = ODEProblem(lockstep_per_ode, u0, tspan, p)

    # PerIndex layout
    lockstep_per_index = LockstepFunction(ode_func, ode_size, num_odes, ordering=PerIndex())
    prob_per_index = ODEProblem(lockstep_per_index, u0, tspan, p)

    println("PerODE layout:")
    @btime solve($prob_per_ode, Tsit5())

    println("\nPerIndex layout:")
    @btime solve($prob_per_index, Tsit5())
end
```

**Typical observations**:
- For small ODE systems (ode_size ≤ 10): Both layouts perform similarly
- For large ODE systems (ode_size > 100): PerODE usually faster
- For many small ODEs with simple dynamics: PerIndex may be faster

---

## Threading Control

LockstepODE uses multi-threading to parallelize execution across different ODEs. You can control this behavior with the `internal_threading` parameter.

### Default Behavior (Threading Enabled)

By default, `internal_threading=true`, and LockstepODE uses `OhMyThreads.jl` to execute ODEs in parallel:

```julia
# Default: threading enabled
lockstep_func = LockstepFunction(my_ode!, 2, 10)  # Solves 10 ODEs in parallel
```

Each ODE system runs on a potentially different thread, allowing efficient use of multi-core processors.

### Disabling Internal Threading

You may want to disable internal threading if:
- You're already parallelizing at a higher level
- You're running in a single-threaded environment
- You want deterministic behavior for debugging

```julia
# Disable internal threading
lockstep_func_no_threading = LockstepFunction(
    my_ode!,
    2,  # 2 variables per ODE
    num_odes;
    internal_threading = false
)

prob_no_threading = ODEProblem(lockstep_func_no_threading, u0_batched, (0.0, 10.0), p_batched)
sol_no_threading = solve(prob_no_threading, Tsit5())
```

With `internal_threading=false`, ODEs are executed sequentially in a simple loop.

### Threading Considerations

#### Thread Safety

When `internal_threading=true`, your ODE function should be thread-safe:

```julia
# Thread-safe: Only modifies local arrays (du, u)
function safe_ode!(du, u, p, t)
    du[1] = p * u[2]
    du[2] = -u[1]
end

# Potentially unsafe: Modifies shared state
global_counter = Ref(0)
function unsafe_ode!(du, u, p, t)
    global_counter[] += 1  # Race condition!
    du[1] = p * u[2]
    du[2] = -u[1]
end
```

If you must modify shared state, use thread-safe primitives:

```julia
using Base.Threads: Atomic, atomic_add!

counter = Atomic{Int}(0)
function thread_safe_counting_ode!(du, u, p, t)
    atomic_add!(counter, 1)  # Thread-safe increment
    du[1] = p * u[2]
    du[2] = -u[1]
end
```

#### Controlling Julia Threads

LockstepODE uses Julia's threading, controlled by the `JULIA_NUM_THREADS` environment variable:

```bash
# Launch Julia with 8 threads
JULIA_NUM_THREADS=8 julia
```

Or from within Julia (before loading packages):

```julia
# Check number of threads
Threads.nthreads()  # Returns: 8 (if launched with 8 threads)
```

### External Parallelization

For nested parallelism (e.g., parallel solves with LockstepODE inside each), disable internal threading:

```julia
using Distributed
@everywhere using LockstepODE, OrdinaryDiffEq

# Create lockstep function without internal threading
@everywhere function setup_problem(param)
    lockstep_func = LockstepFunction(my_ode!, 2, 10, internal_threading=false)
    prob = ODEProblem(lockstep_func, u0, (0.0, 10.0), param)
    return prob
end

# Parallelize across parameter values
params = [0.8, 0.9, 1.0, 1.1, 1.2]
results = pmap(params) do p
    prob = setup_problem(p)
    solve(prob, Tsit5())
end
```

This avoids over-subscription where threads compete for resources.

---

## Performance Tuning

### Benchmarking Your Configuration

For performance-critical applications, benchmark different configurations:

```julia
using BenchmarkTools

function benchmark_configs(ode_func, ode_size, num_odes)
    u0 = ones(ode_size * num_odes)
    p = ones(num_odes)
    tspan = (0.0, 10.0)

    configs = [
        ("PerODE + Threading", PerODE(), true),
        ("PerODE + No Threading", PerODE(), false),
        ("PerIndex + Threading", PerIndex(), true),
        ("PerIndex + No Threading", PerIndex(), false)
    ]

    for (name, ordering, threading) in configs
        lockstep_func = LockstepFunction(
            ode_func, ode_size, num_odes,
            ordering=ordering, internal_threading=threading
        )
        prob = ODEProblem(lockstep_func, u0, tspan, p)

        println("\n$name:")
        @btime solve($prob, Tsit5())
    end
end
```

### General Performance Tips

1. **Use threading**: Keep `internal_threading=true` when solving many (>4) ODEs
2. **Type stability**: Ensure your ODE function is type-stable (check with `@code_warntype`)
3. **Start with PerODE**: Use default `PerODE` layout unless benchmarks show otherwise
4. **Appropriate solver**: Choose ODE solver based on problem stiffness
5. **Batch size**: Performance benefits increase with more ODEs (diminishing returns after ~100)

### Hardware Considerations

**Multi-core processors**: Threading provides significant speedup (near-linear for independent ODEs)

**NUMA systems**: Memory layout may matter more on Non-Uniform Memory Access systems

**SIMD**: PerIndex layout may enable better vectorization for simple operations

---

## Summary

Key configuration options:

| Option | Values | Default | Use Case |
|--------|--------|---------|----------|
| `ordering` | `PerODE()`, `PerIndex()` | `PerODE()` | Memory layout optimization |
| `internal_threading` | `true`, `false` | `true` | Control parallelization |

**Recommended workflow**:
1. Start with defaults (`PerODE`, `internal_threading=true`)
2. If performance is critical, benchmark both layouts
3. Disable threading only for nested parallelism or debugging
4. Use Julia's profiling tools (`@profile`, ProfileView.jl) for detailed analysis

For basic usage patterns, see [Basic Usage](@ref). For event handling, see [Per-ODE Callbacks](@ref).
