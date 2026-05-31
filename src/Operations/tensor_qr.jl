using LinearAlgebra: LinearAlgebra
using ..Muscle: factorinds

# TODO implement low-rank approximations (truncated QR, reduced QR...)

function tensor_qr(A::AbstractArray; inds_q=(), inds_r=(), ind_virtual=Index(gensym(:qr)), inplace=false, kwargs...)
    backend = getbackend(tensor_qr, platform(A))
    return tensor_qr(backend, A; inds_q, inds_r, ind_virtual, inplace, kwargs...)
end

Base.Base.@nospecializeinfer function tensor_qr(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_qr` not implemented or not loaded for backend $backend"))
end

function tensor_qr!(Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    _platform = promote_platform(platform(Q), platform(R), platform(A))
    backend = getbackend(tensor_qr!, _platform)
    return tensor_qr!(backend, Q, R, A; kwargs...)
end

function tensor_qr!(B::Backend, Q::Tensor, R::Tensor, A::Tensor; kwargs...)
    @warn "Fallback to generic `tensor_qr!` implementation for backend $B with intermediate copying."

    tmp_Q, tmp_R = tensor_qr(
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
