module MusclecuTensorNetExt

using CUDA
using cuTensorNet: cuTensorNet
using Muscle
using Muscle: AbsorbBehavior, BackendCuTensorNet
using Base: @nospecializeinfer

function __init__()
    Muscle.register_backend!(BackendCuTensorNet())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.simple_update, BackendCuTensorNet())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.tensor_qr, BackendCuTensorNet())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.tensor_svd, BackendCuTensorNet())
end

# TODO customize SVD algorithm
# TODO configure GPU stream
# TODO cache workspace memory
# TODO do QR before SU to reduce computational cost on A,B with ninds > 3 but not when size(extent) ~ size(rest)
function simple_update(
    ::BackendCuTensorNet,
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray),
    @nospecialize(g::AbstractArray);
    dim_physical_a,
    dim_physical_b,
    dim_bond_a,
    dim_bond_b,
    absorb,
    normalize,
    atol,
    rtol,
    maxdim,
)
    modes_a = collect(1:ndims(a))
    modes_b = ndims(a) .+ (1:ndims(b))
    modes_g = [modes_a[dim_physical_a], modes_b[dim_physical_b], ndims(a) + ndims(b) + 1, ndims(a) + ndims(b) + 2]
    modes_a[dim_bond_a] = ndims(a) + ndims(b) + 3
    modes_b[dim_bond_b] = ndims(a) + ndims(b) + 3
    modes_u = copy(modes_a)
    replace!(modes_u, modes_g[1] => modes_g[3])
    modes_v = copy(modes_b)
    replace!(modes_u, modes_g[2] => modes_g[4])

    # implement maxdim for simple_update on GPU (i think we just need to correctly size U and V beforehand)
    new_size_bond_ab = min(length(A) ÷ size(A, ind_bond_ab), length(B) ÷ size(B, ind_bond_ab))
    if !isnothing(maxdim)
        new_size_bond_ab = min(new_size_bond_ab, maxdim)
    end

    sizes_u = Int[i == dim_bond_a ? new_size_bond_ab : size(a, i) for i in 1:ndims(a)]
    sizes_v = Int[i == dim_bond_b ? new_size_bond_ab : size(b, i) for i in 1:ndims(b)]

    T = Base.promote_eltype(a, b, g)
    u = CUDA.zeros(T, Tuple(sizes_u))
    vt = CUDA.zeros(T, Tuple(sizes_v))
    s = CUDA.zeros(real(T), (size(a, dim_bond_a),))

    svd_config = cuTensorNet.SVDConfig(;
        abs_cutoff=atol,
        rel_cutoff=rtol,
        s_partition=if absorb isa DontAbsorb
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_NONE
        elseif absorb isa AbsorbU
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_US
        elseif absorb isa AbsorbV
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_SV
        elseif absorb isa AbsorbEqually
            cuTensorNet.CUTENSORNET_TENSOR_SVD_PARTITION_UV_EQUAL
        else
            throw(ArgumentError("Unknown value for absorb: $absorb"))
        end,
        s_normalization=if normalize
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_L2
        else
            cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
        end,
    )

    # TODO use svd_info
    _, _, _, svd_info = cuTensorNet.gateSplit!(
        a,
        modes_a,
        b,
        modes_b,
        g,
        modes_g,
        u,
        modes_u,
        s,
        vt,
        modes_v;
        svd_config,
    )

    if absorb isa DontAbsorb
        return u, s, vt
    else
        return u, vt
    end
end

@nospecializeinfer function tensor_qr(::BackendCuTensorNet, @nospecialize(a::AbstractArray); dims, kwargs...)
    modes_q, modes_r = collect.(dims)

    size_q = Int[size(a, i) for i in modes_q]
    size_r = Int[size(a, i) for i in modes_r]
    size_virtual = min(prod(size_q), prod(size_r))

    q = CUDA.zeros(eltype(a), size_q..., size_virtual)
    r = CUDA.zeros(eltype(a), size_virtual, size_r...)

    # call to cuTensorNet QR method is implemented as `LinearAlgebra.qr!`
    LinearAlgebra.qr!(a, collect(1:ndims(a)), q, modes_q, r, modes_r; kwargs...)
    return q, r
end

@nospecializeinfer function tensor_svd(::BackendCuTensorNet, @nospecialize(A::AbstractArray); dims, kwargs...)
    modes_u, modes_v = collect.(dims)

    size_u = Int[size(a, i) for i in modes_u]
    size_v = Int[size(a, i) for i in modes_v]
    size_virtual = min(prod(size_u), prod(size_v))

    u = CUDA.zeros(eltype(a), size_u..., size_virtual)
    s = CUDA.zeros(eltype(a), size_virtual)
    vt = CUDA.zeros(eltype(a), size_virtual, size_v...)

    # call to cuTensorNet SVD method is implemented as `LinearAlgebra.svd!`
    LinearAlgebra.svd!(a, modes_a, u, modes_u, s, v, modes_v; kwargs...)

    return u, s, vt
end

end
