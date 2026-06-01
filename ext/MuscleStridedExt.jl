module MuscleStridedExt

using Muscle
using Muscle: BackendStrided, arraytype
using Strided
using StridedViews
using Base: @nospecializeinfer

function __init__()
    Muscle.register_backend!(BackendStrided())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum, BackendStrided())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum!, BackendStrided())
end

@nospecializeinfer function Muscle.binary_einsum(::BackendStrided, @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims)
    return binary_einsum(
        BackendStrided(),
        arraytype(a) isa StridedView ? a : StridedView(a),
        arraytype(b) isa StridedView ? b : StridedView(b);
        contracting_dims,
        batching_dims,
    )
end

@nospecializeinfer function Muscle.binary_einsum(
    ::BackendStrided, @nospecialize(a::StridedView), @nospecialize(b::StridedView); contracting_dims, batching_dims
)
    @assert isempty(batching_dims) "Batch `binary_einsum` not yet supported for BackendStrided"

    inner_inds_a, inner_inds_b = contracting_dims
    outer_inds_a = Int[i for i in 1:ndims(a) if i ∉ inner_inds_a]
    outer_inds_b = Int[i for i in 1:ndims(b) if i ∉ inner_inds_b]

    sizes_left = Int[size(a, i) for i in outer_inds_a]
    sizes_right = Int[size(b, i) for i in outer_inds_b]
    sizes_c = Tuple(Int[sizes_left; sizes_right])

    c = StridedView(zeros(Base.promote_eltype(a, b), sizes_c), sizes_c)
    binary_einsum!(BackendStrided(), c, a, b; contracting_dims, batching_dims)
    return c
end

@nospecializeinfer function Muscle.binary_einsum!(
    ::BackendStrided, @nospecialize(c::StridedView), @nospecialize(a::StridedView), @nospecialize(b::StridedView)
)
    @assert isempty(batching_dims) "Batch `binary_einsum` not yet supported for BackendStrided"

    inner_inds_a, inner_inds_b = contracting_dims
    outer_inds_a = Int[i for i in 1:ndims(a) if i ∉ inner_inds_a]
    outer_inds_b = Int[i for i in 1:ndims(b) if i ∉ inner_inds_b]

    sizes_left = Int[size(a, i) for i in outer_inds_a]
    sizes_right = Int[size(b, i) for i in outer_inds_b]
    sizes_contract = Int[size(a, i) for i in inner_inds_a]

    a_mat = permutedims(a, [inds_left; inds_contract])
    a_mat = sreshape(a_mat, Tuple([sizes_left; ones(Int, length(sizes_right)); sizes_contract]))

    b_mat = permutedims(b, [inds_right; inds_contract])
    b_mat = sreshape(b_mat, Tuple([ones(Int, length(sizes_left)); sizes_right; sizes_contract]))

    c_mat = permutedims(c, [inds_left; inds_right])
    c_mat = sreshape(c_mat, Tuple([sizes_left; sizes_right; ones(Int, length(sizes_contract))]))

    tsize = (sizes_left..., sizes_right..., sizes_contract...)
    Strided._mapreducedim!(*, +, zero, tsize, (c_mat, a_mat, b_mat))
    return c
end

end
