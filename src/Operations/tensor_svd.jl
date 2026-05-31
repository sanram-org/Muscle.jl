using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated SVD, reduced SVD...)

function tensor_svd! end

function tensor_svd(A::Tensor; inds_u=(), inds_v=(), ind_s=Index(gensym(:svd)), inplace=false, kwargs...)
    backend = getbackend(tensor_svd, platform(A))
    return tensor_svd(backend, A; inds_u, inds_v, ind_s, inplace, kwargs...)
end

Base.@nospecializeinfer function tensor_svd(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_svd` not implemented or not loaded for backend $backend"))
end

function tensor_svd!(U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(U), platform(s), platform(V), platform(A))
    backend = getbackend(tensor_svd!, _platform)
    return tensor_svd!(backend, U, s, V, A; kwargs...)
end

function tensor_svd!(B::Backend, U::Tensor, s::Tensor, V::Tensor, A::Tensor; kwargs...)
    @debug "Fallback to generic `tensor_svd!` implementation for backend $B with intermediate copying."

    tmp_U, tmp_s, tmp_V = tensor_svd(
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
