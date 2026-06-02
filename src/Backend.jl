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
promote_platform_rule(::P, ::P) where {P <: Platform} = P()
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

Base.@nospecializeinfer function available_backends(@nospecialize(op::Function))
    return @lock BACKEND_LOCK get!(AVAILABLE_BACKENDS_FOR_OP, op, Backend[])
end

Base.@nospecializeinfer function available_backends(@nospecialize(op::Function), @nospecialize(platform::Platform))
    return available_backends(platform) ∩ available_backends(op)
end

const AVAILABLE_BACKENDS_FOR_OP = Dict{Function, Vector{Backend}}([
    # unary_einsum => Backend[],
    # unary_einsum! => Backend[],
    binary_einsum => Backend[BackendBase()],
    binary_einsum! => Backend[BackendBase()],
    tensor_qr => Backend[BackendBase()],
    tensor_svd => Backend[BackendBase()],
    tensor_eigen => Backend[BackendBase()],
    simple_update => Backend[BackendBase()],
    # simple_update! => Backend[BackendBase()],
])

const DEFAULT_BACKEND = Dict{Tuple{Function, Platform}, Backend}([
    # unary_einsum
    # (unary_einsum, PlatformHost()) => BackendOMEinsum(),
    # (unary_einsum!, PlatformHost()) => BackendOMEinsum(),
    # (unary_einsum, PlatformCUDA()) => BackendOMEinsum(),
    # (unary_einsum!, PlatformCUDA()) => BackendOMEinsum(),
    # (unary_einsum, PlatformReactant()) => BackendReactant(),
    # (unary_einsum!, PlatformReactant()) => BackendReactant(),

    # binary_einsum
    (binary_einsum, PlatformHost()) => BackendBase(),
    (binary_einsum!, PlatformHost()) => BackendBase(),
    (binary_einsum, PlatformCUDA()) => BackendCuTENSOR(),
    (binary_einsum!, PlatformCUDA()) => BackendCuTENSOR(),
    (binary_einsum, PlatformReactant()) => BackendReactant(),
    (binary_einsum!, PlatformReactant()) => BackendReactant(),
    (binary_einsum, PlatformDagger()) => BackendDagger(),
    (binary_einsum!, PlatformDagger()) => BackendDagger(),

    # tensor_qr
    (tensor_qr, PlatformHost()) => BackendBase(),
    (tensor_qr, PlatformCUDA()) => BackendCuTensorNet(),

    # tensor_svd
    (tensor_svd, PlatformHost()) => BackendBase(),
    (tensor_svd, PlatformCUDA()) => BackendCuTensorNet(),
    (tensor_svd, PlatformReactant()) => BackendReactant(),

    # tensor_eigen
    (tensor_eigen, PlatformHost()) => BackendBase(),

    # simple_update
    (simple_update, PlatformHost()) => BackendBase(),
    # (simple_update!, PlatformHost()) => BackendBase(),
    (simple_update, PlatformCUDA()) => BackendCuTensorNet(),
    # (simple_update!, PlatformCUDA()) => BackendCuTensorNet(),
])

Base.@nospecializeinfer function register_backend_for_op!(@nospecialize(f::Function), @nospecialize(backend::Backend))
    @assert is_backend_available(backend) "Backend $backend is not available. Please register it first using `register_backend!`."
    @lock BACKEND_LOCK push!(get!(AVAILABLE_BACKENDS_FOR_OP, f, Backend[]), backend)
    return nothing
end

Base.@nospecializeinfer function setbackend!(@nospecialize(f::Function), @nospecialize(platform::Platform), @nospecialize(backend::Backend))
    @lock BACKEND_LOCK begin
        @assert is_backend_available(backend) "Backend $backend is not available. Please register it first using `register_backend!`."
        @assert backend ∈ available_backends(f, platform) "Backend $backend does not support function $f on platform $platform."
        DEFAULT_BACKEND[(f, platform)] = backend
    end
    return nothing
end

Base.@nospecializeinfer function getbackend(@nospecialize(f::Function), @nospecialize(platform::Platform))
    return @lock BACKEND_LOCK get(DEFAULT_BACKEND, (f, platform), missing)
end
