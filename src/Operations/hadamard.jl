function hadamard end
function hadamard! end

function hadamard(a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard(b, a)
    @argcheck inds(b) ⊆ inds(a)

    _platform = promote_platform(platform(a), platform(b))
    backend = getbackend(hadamard, _platform)
    return hadamard(backend, a, b)
end

Base.@nospecializeinfer function hadamard(backend::Backend, @nospecialize(a), @nospecialize(b))
    throw(ArgumentError("`hadamard` not implemented or not loaded for backend $backend"))
end

function hadamard!(c::Tensor, a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard(c, b, a)

    @argcheck inds(c) == inds(a)
    @argcheck inds(b) ⊆ inds(a)

    _platform = promote_platform(platform(c), platform(a), platform(b))
    backend = getbackend(hadamard!, _platform)
    return hadamard!(backend, c, a, b)
end

Base.@nospecializeinfer function hadamard!(backend::Backend, @nospecialize(c), @nospecialize(a), @nospecialize(b))
    @debug "Fallback to generic `hadamard!` implementation for backend $backend with intermediate copying."
    _c = hadamard(backend, a, b)
    copyto!(parent(c), parent(_c))
    return c
end
