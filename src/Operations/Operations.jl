module Operations

using ArgCheck

using ..Muscle: Tensor, Index, inds, findperm, nonunique
using ..Muscle: Platform, platform, promote_platform, PlatformHost, PlatformCUDA, PlatformReactant, PlatformDagger
using ..Muscle: Backend, BACKEND_LOCK, is_backend_available
using ..Muscle: BackendBase, BackendOMEinsum, BackendCuTENSOR, BackendCuTensorNet, BackendReactant, BackendDagger
import ..Muscle: available_backends

include("hadamard.jl")
export hadamard, hadamard!

include("unary_einsum.jl")
export unary_einsum, unary_einsum!

include("binary_einsum.jl")
export binary_einsum, binary_einsum!

include("tensor_qr.jl")
export tensor_qr_thin, tensor_qr_thin!

include("tensor_svd.jl")
export tensor_svd_thin, tensor_svd_thin!, tensor_svd_trunc, tensor_svd_trunc!

include("tensor_eigen.jl")
export tensor_eigen_thin, tensor_eigen_thin!
export tensor_bieigen_thin, tensor_bieigen_thin!

include("simple_update.jl")
export simple_update, simple_update!

einsum(a::Tensor; kwargs...) = unary_einsum(a; kwargs...)
einsum(a::Tensor, b::Tensor; kwargs...) = binary_einsum(a, b; kwargs...)
einsum!(c::Tensor, a::Tensor; kwargs...) = unary_einsum!(c, a; kwargs...)
einsum!(c::Tensor, a::Tensor, b::Tensor; kwargs...) = binary_einsum!(c, a, b; kwargs...)
export einsum, einsum!

const AVAILABLE_BACKENDS_FOR_OP = Dict{Function, Vector{Backend}}([
    hadamard => Backend[BackendBase()],
    hadamard! => Backend[BackendBase()],
    unary_einsum => Backend[],
    unary_einsum! => Backend[],
    binary_einsum => Backend[BackendBase()],
    binary_einsum! => Backend[BackendBase()],
    tensor_qr_thin => Backend[BackendBase()],
    tensor_qr_thin! => Backend[BackendBase()],
    tensor_svd_thin => Backend[BackendBase()],
    tensor_svd_thin! => Backend[BackendBase()],
    tensor_svd_trunc => Backend[BackendBase()],
    tensor_svd_trunc! => Backend[BackendBase()],
    tensor_eigen_thin => Backend[BackendBase()],
    tensor_eigen_thin! => Backend[BackendBase()],
    tensor_bieigen_thin => Backend[BackendBase()],
    tensor_bieigen_thin! => Backend[BackendBase()],
    simple_update => Backend[BackendBase()],
    simple_update! => Backend[BackendBase()],
])

const DEFAULT_BACKEND = Dict{Tuple{Function, Platform}, Backend}([
    # hadamard
    (hadamard, PlatformHost()) => BackendBase(),
    (hadamard!, PlatformHost()) => BackendBase(),
    (hadamard, PlatformReactant()) => BackendReactant(),

    # unary_einsum
    (unary_einsum, PlatformHost()) => BackendOMEinsum(),
    (unary_einsum!, PlatformHost()) => BackendOMEinsum(),
    (unary_einsum, PlatformCUDA()) => BackendOMEinsum(),
    (unary_einsum!, PlatformCUDA()) => BackendOMEinsum(),
    (unary_einsum, PlatformReactant()) => BackendReactant(),
    (unary_einsum!, PlatformReactant()) => BackendReactant(),

    # binary_einsum
    (binary_einsum, PlatformHost()) => BackendBase(),
    (binary_einsum!, PlatformHost()) => BackendBase(),
    (binary_einsum, PlatformCUDA()) => BackendCuTENSOR(),
    (binary_einsum!, PlatformCUDA()) => BackendCuTENSOR(),
    (binary_einsum, PlatformReactant()) => BackendReactant(),
    (binary_einsum!, PlatformReactant()) => BackendReactant(),
    (binary_einsum, PlatformDagger()) => BackendDagger(),
    (binary_einsum!, PlatformDagger()) => BackendDagger(),

    # tensor_qr_thin
    (tensor_qr_thin, PlatformHost()) => BackendBase(),
    (tensor_qr_thin!, PlatformHost()) => BackendBase(),
    (tensor_qr_thin, PlatformCUDA()) => BackendCuTensorNet(),
    (tensor_qr_thin!, PlatformCUDA()) => BackendCuTensorNet(),

    # tensor_svd_thin
    (tensor_svd_thin, PlatformHost()) => BackendBase(),
    (tensor_svd_thin!, PlatformHost()) => BackendBase(),
    (tensor_svd_thin, PlatformCUDA()) => BackendCuTensorNet(),
    (tensor_svd_thin!, PlatformCUDA()) => BackendCuTensorNet(),

    # tensor_svd_trunc
    (tensor_svd_trunc, PlatformHost()) => BackendBase(),
    (tensor_svd_trunc!, PlatformHost()) => BackendBase(),
    (tensor_svd_trunc, PlatformCUDA()) => BackendCuTensorNet(),
    (tensor_svd_trunc!, PlatformCUDA()) => BackendCuTensorNet(),

    # tensor_eigen_thin
    (tensor_eigen_thin, PlatformHost()) => BackendBase(),
    (tensor_eigen_thin!, PlatformHost()) => BackendBase(),

    # tensor_bieigen_thin
    (tensor_bieigen_thin, PlatformHost()) => BackendBase(),
    (tensor_bieigen_thin!, PlatformHost()) => BackendBase(),

    # simple_update
    (simple_update, PlatformHost()) => BackendBase(),
    (simple_update!, PlatformHost()) => BackendBase(),
    (simple_update, PlatformCUDA()) => BackendCuTensorNet(),
    (simple_update!, PlatformCUDA()) => BackendCuTensorNet(),
])

Base.@nospecializeinfer function available_backends(@nospecialize(op::Function))
    return @lock BACKEND_LOCK get!(AVAILABLE_BACKENDS_FOR_OP, op, Backend[])
end

Base.@nospecializeinfer function available_backends(@nospecialize(op::Function), @nospecialize(platform::Platform))
    return available_backends(platform) ∩ available_backends(op)
end

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

end
