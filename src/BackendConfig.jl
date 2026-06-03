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
    binary_einsum => Backend[BackendBase()],
    binary_einsum! => Backend[BackendBase()],
    tensor_qr => Backend[BackendBase()],
    tensor_svd => Backend[BackendBase()],
    tensor_eigen => Backend[BackendBase()],
    simple_update => Backend[BackendBase()],
    # simple_update! => Backend[BackendBase()],
])

const DEFAULT_BACKEND = Dict{Tuple{Function, Platform}, Backend}([
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
