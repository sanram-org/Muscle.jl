"""
    binary_einsum(a::Tensor, b::Tensor; dims=∩(inds(a), inds(b)), out=nothing)

Perform a binary tensor contraction operation.

# Keyword arguments

    - `dims`: indices to contract over. Defaults to the set intersection of the indices of `a` and `b`.
    - `out`: indices of the output tensor. Defaults to the set difference of the indices of `a` and `b`.
"""
function binary_einsum end

"""
    binary_einsum!(c::Tensor, a::Tensor, b::Tensor)

Perform a binary tensor contraction operation between `a` and `b` and store the result in `c`.
"""
function binary_einsum! end

function binary_einsum(a::Tensor, b::Tensor; dims=(∩(inds(a), inds(b))), out=nothing)
    inds_sum = ∩(dims, inds(a), inds(b))

    inds_c = if isnothing(out)
        setdiff(inds(a) ∪ inds(b), inds_sum isa Base.AbstractVecOrTuple ? inds_sum : [inds_sum])
    else
        out
    end

    _platform = promote_platform(platform(a), platform(b))
    backend = getbackend(binary_einsum, _platform)
    return binary_einsum(backend, inds_c, a, b)
end

Base.@nospecializeinfer function binary_einsum(@nospecialize(B::Backend), _, @nospecialize(a::Tensor), @nospecialize(b::Tensor))
    throw(ArgumentError("`binary_einsum` not implemented or not loaded for backend $B"))
end

function binary_einsum!(c::Tensor, a::Tensor, b::Tensor)
    _platform = promote_platform(platform(c), platform(a), platform(b))
    backend = getbackend(binary_einsum!, _platform)
    binary_einsum!(backend, c, a, b)
    return c
end

Base.@nospecializeinfer function binary_einsum!(@nospecialize(B::Backend), @nospecialize(c::Tensor), @nospecialize(a::Tensor), @nospecialize(b::Tensor))
    @debug "Fallback to generic `binary_einsum!` implementation for backend $B with intermediate copying."
    _c = binary_einsum(B, inds(c), a, b)
    copyto!(parent(c), parent(_c))
    return c
end

Base.@nospecializeinfer function binary_einsum(::BackendBase, inds_c, @nospecialize(a::Tensor), @nospecialize(b::Tensor))
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices
    @argcheck isdisjoint(inds_c, inds_contract) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."
    @argcheck issetequal(inds_c, symdiff(inds(a), inds(b))) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."

    sizes_left = Int[size(a, ind) for ind in inds_left]
    sizes_right = Int[size(b, ind) for ind in inds_right]
    sizes_contract = Int[size(a, ind) for ind in inds_contract]

    a_mat = reshape(parent(permutedims(a, Index[inds_left; inds_contract])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, Index[inds_contract; inds_right])), prod(sizes_contract), prod(sizes_right))

    c_mat = a_mat * b_mat

    c = Tensor(reshape(c_mat, sizes_left..., sizes_right...), [inds_left; inds_right])
    return permutedims(c, inds_c)
end

Base.@nospecializeinfer function binary_einsum!(::BackendBase, @nospecialize(c::Tensor), @nospecialize(a::Tensor), @nospecialize(b::Tensor))
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices / batching indices
    @argcheck isdisjoint(inds(c), inds_contract) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."
    @argcheck issetequal(inds(c), symdiff(inds(a), inds(b))) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."

    # can't deal with inplace permutedims
    @argcheck inds(c) == [inds_left; inds_right] "`BackendBase` can't deal with inplace permutedims. Load OMEinsum and use `BackendOMEinsum` instead."

    sizes_left = Int[size(a, ind) for ind in inds_left]
    sizes_right = Int[size(b, ind) for ind in inds_right]
    sizes_contract = Int[size(a, ind) for ind in inds_contract]

    a_mat = reshape(parent(permutedims(a, Index[inds_left; inds_contract])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, Index[inds_contract; inds_right])), prod(sizes_contract), prod(sizes_right))
    c_mat = reshape(c, prod(sizes_left), prod(sizes_right))

    LinearAlgebra.mul!(c_mat, a_mat, b_mat)

    return reshape(c_mat, sizes_left..., sizes_right...)
end
