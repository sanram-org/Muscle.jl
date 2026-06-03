module MuscleOMEinsumExt

using Muscle
using Muscle: BackendOMEinsum
using OMEinsum
using Base: @nospecializeinfer

function __init__()
    Muscle.register_backend!(BackendOMEinsum())
    Muscle.register_backend_for_op!(Muscle.binary_einsum, BackendOMEinsum())
    Muscle.register_backend_for_op!(Muscle.binary_einsum!, BackendOMEinsum())
end

@nospecializeinfer function Muscle.binary_einsum(
    ::Muscle.BackendOMEinsum,
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    inner_dims_a, inner_dims_b = collect.(contracting_dims)
    batch_dims_a, batch_dims_b = collect.(batching_dims)
    outer_dims_a = Int[i for i in 1:ndims(a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(b) if i ∉ inner_dims_b && i ∉ batch_dims_b]

    csize = Int[
        Int[size(a, d) for d in batch_dims_a]
        Int[size(a, d) for d in outer_dims_a]
        Int[size(b, d) for d in outer_dims_b]
    ]

    c = OMEinsum.get_output_array((a, b), csize, false)
    Muscle.binary_einsum!(Muscle.BackendOMEinsum(), c, a, b; contracting_dims, batching_dims)
    return c
end

@nospecializeinfer function Muscle.binary_einsum!(
    ::Muscle.BackendOMEinsum,
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    inner_dims_a, inner_dims_b = collect.((Int,), contracting_dims)
    batch_dims_a, batch_dims_b = collect.((Int,), batching_dims)
    outer_dims_a = Int[i for i in 1:ndims(a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(b) if i ∉ inner_dims_b && i ∉ batch_dims_b]

    n_inner = length(inner_dims_a)
    n_batch = length(batch_dims_a)
    inner_inds = collect(Int, 1:n_inner)
    batch_inds = collect(Int, n_inner .+ (1:n_batch))
    outer_inds_a = collect(Int, n_inner + n_batch .+ (1:length(outer_dims_a)))
    outer_inds_b = collect(Int, n_inner + n_batch + length(outer_inds_a) .+ (1:length(outer_dims_b)))

    inds_a = Vector{Int}(undef, ndims(a))
    for d in 1:ndims(a)
        i = findfirst(==(d), inner_dims_a)
        !isnothing(i) && (inds_a[d] = inner_inds[i])

        i = findfirst(==(d), batch_dims_a)
        !isnothing(i) && (inds_a[d] = batch_inds[i])

        i = findfirst(==(d), outer_dims_a)
        !isnothing(i) && (inds_a[d] = outer_inds_a[i])
    end

    inds_b = Vector{Int}(undef, ndims(b))
    for d in 1:ndims(b)
        i = findfirst(==(d), inner_dims_b)
        !isnothing(i) && (inds_b[d] = inner_inds[i])

        i = findfirst(==(d), batch_dims_b)
        !isnothing(i) && (inds_b[d] = batch_inds[i])

        i = findfirst(==(d), outer_dims_b)
        !isnothing(i) && (inds_b[d] = outer_inds_b[i])
    end

    inds_c = Int[batch_inds; outer_inds_a; outer_inds_b]

    code = OMEinsum.DynamicEinCode([inds_a, inds_b], inds_c)
    OMEinsum.einsum!(code, (a, b), c, true, false)
    return c
end

end
