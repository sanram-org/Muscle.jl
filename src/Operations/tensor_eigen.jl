using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated eigen, reduced eigen...)

"""
    Muscle.tensor_eigen_thin(tensor::Tensor; inds_u, inds_uinv, ind_lambda, kwargs...)

Perform eigen factorization on a tensor. Either `inds_u` or `inds_uinv` must be specified.

# Keyword arguments

  - `inds_u`: left / U indices to be used in the eigen factorization, except for `ind_lambda`.
  - `inds_uinv`: right / right indices to be used in the eigen factorization, except for `ind_lambda`.
  - `ind_lambda`: name of the virtual bond.
  - `inplace`: If `true`, it will use `A` as workspace variable to save space. Defaults to `false`.
  - `kwargs...`: additional keyword arguments to be passed to `LinearAlgebra.eigen`.
"""
function tensor_eigen_thin end
function tensor_eigen_thin! end

function tensor_bieigen_thin end
function tensor_bieigen_thin! end

function tensor_eigen_thin(
    A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:eigen)), inplace=false, kwargs...
)
    backend = getbackend(tensor_eigen_thin, platform(A))
    return tensor_eigen_thin(backend, A; inds_u, inds_uinv, ind_lambda, inplace, kwargs...)
end

function tensor_bieigen_thin(
    A::Tensor; inds_u=(), inds_uinv=(), ind_lambda=Index(gensym(:eigen)), inplace=false, kwargs...
)
    backend = getbackend(tensor_bieigen_thin, platform(A))
    return tensor_bieigen_thin(backend, A; inds_u, inds_uinv, ind_lambda, inplace, kwargs...)
end

function tensor_eigen_thin!(B::Backend, Λ::Tensor, U::Tensor, A::Tensor; kwargs...)
    @warn "Fallback to generic `tensor_eigen_thin!` implementation for backend $B with intermediate copying."

    tmp_Λ, tmp_U = tensor_eigen_thin(B, A; inds_u=inds(U), ind_lambda=only(inds(Λ)), kwargs...)

    @argcheck inds(tmp_Λ) == inds(Λ)
    @argcheck inds(tmp_U) == inds(U)

    @argcheck size(tmp_Λ) == size(Λ)
    @argcheck size(tmp_U) == size(U)

    copyto!(Λ, tmp_Λ)
    copyto!(U, tmp_U)

    return Λ, U
end

function tensor_bieigen_thin!(B::Backend, Uinv::Tensor, Λ::Tensor, U::Tensor, A::Tensor; kwargs...)
    @warn "Fallback to generic `tensor_bieigen_thin!` implementation for backend $B with intermediate copying."

    tmp_Uinv, tmp_Λ, tmp_U = tensor_bieigen_thin(
        B, A; inds_u=inds(U), inds_uinv=inds(Uinv), ind_lambda=only(inds(Λ)), kwargs...
    )

    @argcheck inds(tmp_Uinv) == inds(Uinv)
    @argcheck inds(tmp_Λ) == inds(Λ)
    @argcheck inds(tmp_U) == inds(U)

    @argcheck size(tmp_Uinv) == size(Uinv)
    @argcheck size(tmp_Λ) == size(Λ)
    @argcheck size(tmp_U) == size(U)

    copyto!(Uinv, tmp_Uinv)
    copyto!(Λ, tmp_Λ)
    copyto!(U, tmp_U)

    return Uinv, Λ, U
end
