function tensor_svd_thin(
    ::BackendBase, A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
    @argcheck isdisjoint(inds_u, inds_v)
    @argcheck issetequal(inds_u ∪ inds_v, inds(A))
    @argcheck ind_s ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_u)
    right_sizes = map(Base.Fix1(size, A), inds_v)
    Amat = permutedims(A, [inds_u; inds_v])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute SVD
    F = if inplace
        LinearAlgebra.svd!(Amat; kwargs...)
    else
        LinearAlgebra.svd(Amat; kwargs...)
    end

    # tensorify results
    U = Tensor(reshape(F.U, left_sizes..., size(F.U, 2)), [inds_u; ind_s])
    s = Tensor(F.S, [ind_s])
    Vt = Tensor(reshape(F.Vt, size(F.Vt, 1), right_sizes...), [ind_s; inds_v])

    return U, s, Vt
end

# TODO implement for cuTensorNet
"""
Truncate SVD. With these defaults, it could be used as inplace replacement for tensor_svd_thin
"""
function tensor_svd_trunc(
    ::BackendBase,
    A::Tensor;
    inds_u=(),
    inds_v=(),
    ind_s=Index(gensym(:vind)),
    inplace=false,
    threshold=nothing,
    maxdim=nothing,
    kwargs...,
)
    inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
    @argcheck isdisjoint(inds_u, inds_v)
    @argcheck issetequal(inds_u ∪ inds_v, inds(A))
    @argcheck ind_s ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_u)
    right_sizes = map(Base.Fix1(size, A), inds_v)
    Amat = permutedims(A, [inds_u..., inds_v...])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute SVD
    F = if inplace
        LinearAlgebra.svd!(Amat; kwargs...)
    else
        LinearAlgebra.svd(Amat; kwargs...)
    end

    # truncate singular values
    k = length(F.S)

    # use `maxdim` to truncate the singular values
    if !isnothing(maxdim)
        k = min(k, maxdim)
    end

    # use `threshold` to truncate the singular values
    if !isnothing(threshold)
        # threshold is relative threshold
        threshold = LinearAlgebra.norm(F.S) * threshold
        k = something(findfirst(<(threshold), view(F.S, 1:k)), k)
    end

    keep = 1:k

    view_u = view(F.U, :, keep)
    view_s = view(F.S, keep)
    view_v = view(F.Vt, keep, :)

    # tensorify results
    U = Tensor(reshape(view_u, left_sizes..., k), [inds_u; ind_s])
    s = Tensor(view_s, [ind_s])
    Vt = Tensor(reshape(view_v, k, right_sizes...), [ind_s; inds_v])

    return U, s, Vt
end
