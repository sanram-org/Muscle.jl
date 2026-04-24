using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated QR, reduced QR...)

"""
    tensor_qr_thin(tensor::Tensor; inds_q, inds_r, ind_virtual, kwargs...)

Perform QR factorization on a tensor. Either `inds_q` or `inds_r` must be specified.

# Keyword arguments

  - `inds_q`: left indices to be used in the QR factorization. Defaults to all indices of `t` except `inds_r`.
  - `inds_r`: right indices to be used in the QR factorization. Defaults to all indices of `t` except `inds_q`.
  - `ind_virtual`: name of the virtual bond. Defaults to a random `Index`.
"""
function tensor_qr_thin end
function tensor_qr_thin! end

function tensor_qr_thin(A::Tensor; inds_q=(), inds_r=(), ind_virtual=Index(gensym(:qr)), inplace=false, kwargs...)
    backend = getbackend(tensor_qr_thin, platform(A))
    return tensor_qr_thin(backend, A; inds_q, inds_r, ind_virtual, inplace, kwargs...)
end

Base.Base.@nospecializeinfer function tensor_qr_thin(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_qr_thin` not implemented or not loaded for backend $backend"))
end

function tensor_qr_thin!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(Q), platform(R), platform(A))
    backend = getbackend(tensor_qr_thin!, _platform)
    return tensor_qr_thin!(backend, Q, R, A; kwargs...)
end

function tensor_qr_thin!(B::Backend, Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    @warn "Fallback to generic `tensor_qr_thin!` implementation for backend $B with intermediate copying."

    tmp_Q, tmp_R = tensor_qr_thin(
        B, A; inds_q=setdiff(inds(Q), inds(R)), inds_r=setdiff(inds(R), inds(Q)), kwargs...
    )

    @argcheck inds(tmp_Q) == inds(Q)
    @argcheck inds(tmp_R) == inds(R)

    @argcheck size(tmp_Q) == size(Q)
    @argcheck size(tmp_R) == size(R)

    copyto!(Q, tmp_Q)
    copyto!(R, tmp_R)

    return Q, R
end
