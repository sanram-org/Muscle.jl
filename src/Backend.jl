using ScopedValues

abstract type Platform end

struct PlatformHost <: Platform end
struct PlatformCUDA <: Platform end
struct PlatformReactant <: Platform end
struct PlatformDagger <: Platform end

platform(x::AbstractArray) = platform(parent(x))
platform(::Array) = PlatformHost()

function promote_platform(a::Platform, b::Platform)
    ab = promote_platform_rule(a, b)
    ba = promote_platform_rule(b, a)
    res = promote_platform_result(ab, ba)
    if ismissing(res)
        throw(ArgumentError("No promotion found for $a and $b"))
    else
        return res
    end
end

promote_platform(a, b, c, args...) = promote_platform(promote_platform(a, b), c, args...)
promote_platform(a::AbstractArray, b::AbstractArray) = promote_platform(Platform(a), Platform(b))

promote_platform_result(::Missing, ::Missing) = missing
promote_platform_result(@nospecialize(ab::Platform), ::Missing) = ab
promote_platform_result(::Missing, @nospecialize(ba::Platform)) = ba
promote_platform_result(@nospecialize(ab::Platform), @nospecialize(ba::Platform)) = (@assert(ab == ba); ab)

promote_platform_rule(@nospecialize(::Platform), @nospecialize(::Platform)) = missing
promote_platform_rule(::P, ::P) where {P<:Platform} = P()
promote_platform_rule(::PlatformHost, ::PlatformCUDA) = PlatformCUDA()
promote_platform_rule(::PlatformHost, ::PlatformReactant) = PlatformReactant()
promote_platform_rule(::PlatformHost, ::PlatformDagger) = PlatformDagger()
promote_platform_rule(::PlatformCUDA, ::PlatformReactant) = PlatformReactant()

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
