using ArgCheck
using ScopedValues

"""
    Backend

Abstract type representing a computational backend for tensor operations.
Current backends include:

- `BackendBase` using purely Base and/or LinearAlgebra stdlibs.
- `BackendStrided` using [Strided.jl](https://github.com/QuantumKitHub/Strided.jl).
- `BackendOMEinsum` using the [OMEinsum.jl](https://github.com/under-peter/OMEinsum.jl).
- `BackendCUDA` using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
- `BackendCuTENSOR` using [CuTENSOR.jl](https://github.com/JuliaGPU/CUDA.jl/tree/master/lib/cutensor).
- `BackendCuTensorNet` using [CuTensorNet.jl](https://github.com/JuliaGPU/CUDA.jl/tree/master/lib/cutensornet).
- `BackendReactant` using [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl).
- `BackendDagger` using [Dagger.jl](https://github.com/JuliaParallel/Dagger.jl).
"""
abstract type Backend end

struct BackendBase <: Backend end
struct BackendStrided <: Backend end
struct BackendOMEinsum <: Backend end
struct BackendCUDA <: Backend end
struct BackendCuTENSOR <: Backend end
struct BackendCuTensorNet <: Backend end
struct BackendReactant <: Backend end
struct BackendDagger <: Backend end

supported_platforms(::BackendBase) = Platform[PlatformHost()]
supported_platforms(::BackendStrided) = Platform[PlatformHost(), PlatformCUDA()]
supported_platforms(::BackendOMEinsum) = Platform[PlatformHost(), PlatformCUDA()]
supported_platforms(::BackendCUDA) = Platform[PlatformCUDA()]
supported_platforms(::BackendCuTENSOR) = Platform[PlatformCUDA()]
supported_platforms(::BackendCuTensorNet) = Platform[PlatformCUDA()]
supported_platforms(::BackendReactant) = Platform[PlatformReactant()]
supported_platforms(::BackendDagger) = Platform[PlatformDagger()]

const BACKEND_LOCK = ReentrantLock()
const AVAILABLE_BACKENDS = Set{Backend}([BackendBase()])

Base.@nospecializeinfer function register_backend!(@nospecialize(backend::Backend))
    @lock BACKEND_LOCK push!(AVAILABLE_BACKENDS, backend)
    return nothing
end

Base.@nospecializeinfer function is_backend_available(@nospecialize(backend::Backend))
    return backend ∈ AVAILABLE_BACKENDS
end

"""
    available_backends()

Returns the set of all currently available [`Backend`](@ref)s.
"""
available_backends() = copy(AVAILABLE_BACKENDS)

"""
    available_backends(platform::Platform)

Returns the set of available [`Backend`](@ref)s that support the given [`Platform`](@ref).
"""
Base.@nospecializeinfer function available_backends(@nospecialize(platform::Platform))
    return filter(backend -> platform in supported_platforms(backend), AVAILABLE_BACKENDS)
end
