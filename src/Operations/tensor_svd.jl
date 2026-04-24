using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated SVD, reduced SVD...)

"""
    Muscle.tensor_svd_thin(tensor::Tensor; inds_u, inds_v, ind_s, kwargs...)

Perform SVD factorization on a tensor. Either `inds_u` or `inds_v` must be specified.

# Keyword arguments

  - `inds_u`: left / U indices to be used in the SVD factorization, except for `ind_s`.
  - `inds_v`: right / right indices to be used in the SVD factorization, except for `ind_s`.
  - `ind_s`: name of the virtual bond.
  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.svd`.
"""
function tensor_svd_thin end
function tensor_svd_thin! end

"""
    Muscle.tensor_svd_trunc(tensor::Tensor; inds_u, inds_v, ind_s, threshold, maxbondim, kwargs...)

Same as tensor_svd_thin() but with truncation, additional keyword arguments are

  - `threshold`: relative cutoff for singular values
  - `maxdim`: maximum bond dimension
"""
function tensor_svd_trunc end
function tensor_svd_trunc! end

# function allocate_result(::typeof(tensor_svd_thin), A; inds_u=(), inds_v=(), ind_s=Index(gensym(:s)), kwargs...)
#     inds_u, inds_v = factorinds(inds(A), inds_u, inds_v)
#     left_extent = prod(Base.Fix1(size, A), inds_u)
#     right_extent = prod(Base.Fix1(size, A), inds_v)
#     s_extent = min(left_extent, right_extent)

#     U = Tensor(similar(parent(A), left_extent..., s_extent), [inds_u..., ind_s])
#     s = Tensor(similar(parent(A), s_extent), [ind_s])
#     V = Tensor(similar(parent(A), right_extent..., s_extent), [inds_v..., ind_s])

#     return U, s, V
# end

# function tensor_svd_thin!(A::Tensor, U::Tensor, s::Tensor, V::Tensor; kwargs...)
#     @argcheck arch(A) == arch(U) == arch(s) == arch(V)
#     tensor_svd_thin!(arch(A), A, U, s, V; kwargs...)
# end

function tensor_svd_thin(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:svd)), inplace=false, kwargs...)
    backend = getbackend(tensor_svd_thin, platform(A))
    return tensor_svd_thin(backend, A; inds_u, inds_v, ind_s, inplace, kwargs...)
end

Base.@nospecializeinfer function tensor_svd_thin(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_svd_thin` not implemented or not loaded for backend $backend"))
end

function tensor_svd_thin!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(Q), platform(R), platform(A))
    backend = getbackend(tensor_svd_thin!, _platform)
    return tensor_svd_thin!(backend, Q, R, A; kwargs...)
end

Base.@nospecializeinfer function tensor_svd_thin!(backend::Backend, @nospecialize(args...); kwargs...)
    throw(ArgumentError("`tensor_svd_thin!` not implemented or not loaded for backend $backend"))
end

function tensor_svd_trunc(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:svd)), inplace=false, kwargs...)
    backend = getbackend(tensor_svd_trunc, platform(A))
    return tensor_svd_trunc(backend, A; inds_u, inds_v, ind_s, inplace, kwargs...)
end

Base.@nospecializeinfer function tensor_svd_trunc(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_svd_trunc` not implemented or not loaded for backend $backend"))
end

function tensor_svd_trunc!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(Q), platform(R), platform(A))
    backend = getbackend(tensor_svd_trunc!, _platform)
    return tensor_svd_trunc!(backend, Q, R, A; kwargs...)
end

Base.@nospecializeinfer function tensor_svd_trunc!(backend::Backend, @nospecialize(args...); kwargs...)
    throw(ArgumentError("`tensor_svd_trunc!` not implemented or not loaded for backend $backend"))
end

## `Base`
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

function tensor_svd_thin!(B::Backend, U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    @debug "Fallback to generic `tensor_svd_thin!` implementation for backend $B with intermediate copying."

    tmp_U, tmp_s, tmp_V = tensor_svd_thin(
        B, A; inds_u=inds(U), inds_v=inds(V), ind_s=only(inds(s)), kwargs...
    )

    @argcheck inds(tmp_U) == inds(U)
    @argcheck inds(tmp_s) == inds(s)
    @argcheck inds(tmp_V) == inds(V)

    @argcheck size(tmp_U) == size(U)
    @argcheck size(tmp_s) == size(s)
    @argcheck size(tmp_V) == size(V)

    copyto!(U, tmp_U)
    copyto!(s, tmp_s)
    copyto!(V, tmp_V)

    return U, s, V
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

function tensor_svd_trunc!(::BackendBase, U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    @warn "tensor_svd_trunc! on BackendBase does intermediate copying. Consider using `tensor_svd_trunc`."

    tmp_U, tmp_s, tmp_V = tensor_svd_trunc(
        BackendBase(), A; inds_u=inds(U), inds_v=inds(V), ind_s=only(inds(s)), kwargs...
    )

    @argcheck arch(tmp_U) == arch(U)
    @argcheck arch(tmp_s) == arch(s)
    @argcheck arch(tmp_V) == arch(V)

    @argcheck inds(tmp_U) == inds(U)
    @argcheck inds(tmp_s) == inds(s)
    @argcheck inds(tmp_V) == inds(V)

    @argcheck size(tmp_U) == size(U)
    @argcheck size(tmp_s) == size(s)
    @argcheck size(tmp_V) == size(V)

    copyto!(U, tmp_U)
    copyto!(s, tmp_s)
    copyto!(V, tmp_V)

    return U, s, V
end
