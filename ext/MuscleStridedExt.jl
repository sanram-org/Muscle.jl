module MuscleStridedExt

using Muscle
using Muscle: BackendStrided
using Strided
using StridedViews
using Base: @nospecializeinfer

function __init__()
    Muscle.register_backend!(BackendStrided())
    Muscle.register_backend_for_op!(Muscle.binary_einsum, BackendStrided())
    Muscle.register_backend_for_op!(Muscle.binary_einsum!, BackendStrided())
end

@nospecializeinfer function Muscle.binary_einsum(
    ::BackendStrided, @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims
)
    return Muscle.binary_einsum(
        BackendStrided(),
        a isa StridedView ? a : StridedView(a),
        b isa StridedView ? b : StridedView(b);
        contracting_dims,
        batching_dims,
    )
end

@nospecializeinfer function Muscle.binary_einsum(
    ::BackendStrided, @nospecialize(a::StridedView), @nospecialize(b::StridedView); contracting_dims, batching_dims
)
    @assert all(isempty, batching_dims) "Batch `binary_einsum` not yet supported for BackendStrided"

    inner_dims_a, inner_dims_b = contracting_dims
    outer_dims_a = Int[i for i in 1:ndims(a) if i ∉ inner_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(b) if i ∉ inner_dims_b]

    sizes_outer_a = Int[size(a, i) for i in outer_dims_a]
    sizes_outer_b = Int[size(b, i) for i in outer_dims_b]
    sizes_c = Tuple(Int[sizes_outer_a; sizes_outer_b])

    c = StridedView(zeros(Base.promote_eltype(a, b), sizes_c), sizes_c)
    Muscle.binary_einsum!(BackendStrided(), c, a, b; contracting_dims, batching_dims)
    return c
end

@nospecializeinfer function Muscle.binary_einsum!(
    ::BackendStrided,
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    return Muscle.binary_einsum!(
        BackendStrided(),
        c isa StridedView ? c : StridedView(c),
        a isa StridedView ? a : StridedView(a),
        b isa StridedView ? b : StridedView(b);
        contracting_dims,
        batching_dims,
    )
end

@nospecializeinfer function Muscle.binary_einsum!(
    ::BackendStrided,
    @nospecialize(c::StridedView),
    @nospecialize(a::StridedView),
    @nospecialize(b::StridedView);
    contracting_dims,
    batching_dims,
)
    @assert all(isempty, batching_dims) "Batch `binary_einsum` not yet supported for BackendStrided"

    inner_dims_a, inner_dims_b = collect.((Int,), contracting_dims)
    outer_dims_a = Int[i for i in 1:ndims(a) if i ∉ inner_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(b) if i ∉ inner_dims_b]

    sizes_outer_a = Int[size(a, i) for i in outer_dims_a]
    sizes_outer_b = Int[size(b, i) for i in outer_dims_b]
    sizes_inner = Int[size(a, i) for i in inner_dims_a]

    a_mat = permutedims(a, Int[outer_dims_a; inner_dims_a])
    a_mat = sreshape(a_mat, Tuple([sizes_outer_a; ones(Int, length(sizes_outer_b)); sizes_inner]))

    b_mat = permutedims(b, Int[outer_dims_b; inner_dims_b])
    b_mat = sreshape(b_mat, Tuple([ones(Int, length(sizes_outer_a)); sizes_outer_b; sizes_inner]))

    c_mat = sreshape(c, Tuple([sizes_outer_a; sizes_outer_b; ones(Int, length(sizes_inner))]))

    tsize = (sizes_outer_a..., sizes_outer_b..., sizes_inner...)
    Strided._mapreducedim!(*, +, zero, tsize, (c_mat, a_mat, b_mat))
    return c
end

end
