Base.@nospecializeinfer function binary_einsum(::BackendBase, inds_c, @nospecialize(a::Tensor), @nospecialize(b::Tensor))
    inds_contract = inds(a) ∩ inds(b)
    inds_left = setdiff(inds(a), inds_contract)
    inds_right = setdiff(inds(b), inds_contract)

    # can't deal with hyperindices
    @assert isdisjoint(inds_c, inds_contract) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."
    @assert issetequal(inds_c, symdiff(inds(a), inds(b))) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."

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
    @assert isdisjoint(inds(c), inds_contract) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."
    @assert issetequal(inds(c), symdiff(inds(a), inds(b))) "`BackendBase` can't deal with batching indices. Load OMEinsum and use `BackendOMEinsum` instead."

    # can't deal with inplace permutedims
    @assert inds(c) == [inds_left; inds_right] "`BackendBase` can't deal with inplace permutedims. Load OMEinsum and use `BackendOMEinsum` instead."

    sizes_left = Int[size(a, ind) for ind in inds_left]
    sizes_right = Int[size(b, ind) for ind in inds_right]
    sizes_contract = Int[size(a, ind) for ind in inds_contract]

    a_mat = reshape(parent(permutedims(a, Index[inds_left; inds_contract])), prod(sizes_left), prod(sizes_contract))
    b_mat = reshape(parent(permutedims(b, Index[inds_contract; inds_right])), prod(sizes_contract), prod(sizes_right))
    c_mat = reshape(c, prod(sizes_left), prod(sizes_right))

    LinearAlgebra.mul!(c_mat, a_mat, b_mat)

    return reshape(c_mat, sizes_left..., sizes_right...)
end
