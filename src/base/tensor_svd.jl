function tensor_svd(
    ::BackendBase, A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
    @assert isdisjoint(inds_u, inds_v)
    @assert issetequal(inds_u ∪ inds_v, inds(A))
    @assert ind_s ∉ inds(A)

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
