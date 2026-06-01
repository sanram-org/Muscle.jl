function tensor_eigen(
    ::BackendBase, A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_uinv = factorinds(inds(A), inds_u, inds_uinv)
    @assert isdisjoint(inds_u, inds_uinv)
    @assert issetequal(inds_u ∪ inds_uinv, inds(A))
    @assert ind_lambda ∉ inds(A)

    # permute array
    left_sizes = map(Base.Fix1(size, A), inds_uinv)
    right_sizes = map(Base.Fix1(size, A), inds_u)

    Amat = permutedims(A, [inds_u; inds_uinv])
    Amat = reshape(parent(Amat), prod(left_sizes), prod(right_sizes))

    # compute eigen
    F = if inplace
        LinearAlgebra.eigen!(Amat; kwargs...)
    else
        LinearAlgebra.eigen(Amat; kwargs...)
    end

    # tensorify results
    Λ = Tensor(F.values, [ind_lambda])
    #U = Tensor(reshape(F.vectors, size(F.vectors, 2), right_sizes...), [inds_u; ind_lambda])
    U = Tensor(reshape(F.vectors, left_sizes..., size(F.vectors, 2)), [inds_u; ind_lambda])

    return Λ, U
end

# TODO how can we get the left_sizes for the Uinv matrix from `U`?
function tensor_bieigen(
    ::BackendBase, A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:vind)), inplace=false, kwargs...
)
    inds_u, inds_uinv = factorinds(inds(A), inds_u, inds_uinv)
    Λ, U = tensor_eigen(BackendBase(), A; inds_u, inds_uinv, ind_lambda, inplace, kwargs...)

    Umat = reshape(parent(U), size(U, ind_lambda), prod(map(Base.Fix1(size, U), inds_u)))

    left_sizes = map(Base.Fix1(size, A), inds_uinv)

    # TODO probably a better way to get the `Uinv` matrix
    Uinv = Tensor(reshape(inv(Umat), left_sizes..., size(Λ, ind_lambda)), [ind_lambda; inds_uinv])

    return U, Λ, Uinv
end
