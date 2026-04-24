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

function tensor_svd_thin(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:svd)), inplace=false, kwargs...)
    backend = getbackend(tensor_svd_thin, platform(A))
    return tensor_svd_thin(backend, A; inds_u, inds_v, ind_s, inplace, kwargs...)
end

Base.@nospecializeinfer function tensor_svd_thin(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_svd_thin` not implemented or not loaded for backend $backend"))
end

function tensor_svd_thin!(U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(U), platform(s), platform(V), platform(A))
    backend = getbackend(tensor_svd_thin!, _platform)
    return tensor_svd_thin!(backend, U, s, V, A; kwargs...)
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

function tensor_svd_trunc(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:svd)), inplace=false, kwargs...)
    backend = getbackend(tensor_svd_trunc, platform(A))
    return tensor_svd_trunc(backend, A; inds_u, inds_v, ind_s, inplace, kwargs...)
end

Base.@nospecializeinfer function tensor_svd_trunc(backend::Backend, @nospecialize(A::Tensor); maxdim=nothing, threshold=nothing, kwargs...)
    @debug "Fallback to generic `tensor_svd_trunc` implementation for backend $backend with call to `tensor_svd_thin`."
    U, S, Vt = tensor_svd_thin(backend, A; kwargs...)
    ind_s = only(inds(S))

    # truncate singular values
    k = length(S)

    # use `maxdim` to truncate the singular values
    if !isnothing(maxdim)
        k = min(k, maxdim)
    end

    # use `threshold` to truncate the singular values
    if !isnothing(threshold)
        # threshold is relative threshold
        threshold = LinearAlgebra.norm(S) * threshold
        k = something(findfirst(<(threshold), view(S, ind_s => 1:k)), k)
    end

    view_u = view(U, ind_s => 1:k)
    view_s = view(S, ind_s => 1:k)
    view_vt = view(Vt, ind_s => 1:k)

    return view_u, view_s, view_vt
end

function tensor_svd_trunc!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(Q), platform(R), platform(A))
    backend = getbackend(tensor_svd_trunc!, _platform)
    return tensor_svd_trunc!(backend, Q, R, A; kwargs...)
end

function tensor_svd_trunc!(B::Backend, U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    @warn "Fallback to generic `tensor_svd_trunc!` implementation for backend $B with intermediate copying."

    tmp_U, tmp_s, tmp_V = tensor_svd_trunc(
        BackendBase(), A; inds_u=inds(U), inds_v=inds(V), ind_s=only(inds(s)), kwargs...
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
