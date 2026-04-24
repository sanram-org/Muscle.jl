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
    @debug "Fallback to generic `hadamard` implementation for backend $backend"
    _c = hadamard(backend, a, b)
    copyto!(parent(c), parent(_c))
    return c
end

function hadamard(::BackendBase, a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard(BackendBase(), b, a)

    c = copy(a)
    return hadamard!(BackendBase(), c, a, b)
end

function hadamard!(::BackendBase, c::Tensor, a::Tensor, b::Tensor)
    # `b` must be broadcastable to `a`
    ndims(a) >= ndims(b) || return hadamard!(BackendBase(), c, b, a)

    @argcheck inds(c) == inds(a)

    # check if this is just a tensor-scalar multiplication
    if ndims(b) == 0
        c .= a .* b
        return c
    end

    data_a = parent(a)
    data_b = parent(b)
    data_c = parent(c)

    # compute the broadcast shape for `b`
    shape_b_bcast = ones(Int, ndims(a))
    for (i, ind) in enumerate(inds(a))
        if ind ∈ inds(b)
            shape_b_bcast[i] = size(b, ind)
        end
    end

    # if `b` is not a vector, it may need permutation to have the inds in the same order as `a`
    if ndims(b) > 1
        perm = findperm(inds(b), filter(∈(inds(b)), inds(a)))
        data_b = permutedims(data_b, perm)
    end

    # broadcast element-wise multiplication (Hadamard product)
    data_b = reshape(data_b, Tuple(shape_b_bcast))
    data_c .= data_a .* data_b

    return c
end
