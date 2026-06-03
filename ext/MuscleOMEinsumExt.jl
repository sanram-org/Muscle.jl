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
    outer_dims_a = Int[size(a, i) for i in 1:ndims(a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[size(b, i) for i in 1:ndims(b) if i ∉ inner_dims_b && i ∉ batch_dims_b]

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
    inner_dims_a, inner_dims_b = collect.(contracting_dims)
    batch_dims_a, batch_dims_b = collect.(batching_dims)
    outer_dims_a = Int[size(a, i) for i in 1:ndims(a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[size(b, i) for i in 1:ndims(b) if i ∉ inner_dims_b && i ∉ batch_dims_b]

    n_inner = length(inner_dims_a)
    n_batch = length(batch_dims_a)
    inner_inds = collect(1:n_inner)
    batch_inds = collect(n_inner .+ (1:n_batch))
    outer_inds_a = collect(n_inner + n_batch .+ (1:length(outer_dims_a)))
    outer_inds_b = collect(n_inner + n_batch .+ (1:length(outer_dims_b)))

    inds_a = map(1:ndims(a)) do d
        i = findfirst(==(d), inner_dims_a)
        !isnothing(i) && return inner_inds[i]

        i = findfirst(==(d), batch_dims_a)
        !isnothing(i) && return batch_inds[i]

        i = findfirst(==(d), inner_dims_a)
        !isnothing(i) && return outer_inds_a[i]
    end::Vector{Int}

    inds_b = map(1:ndims(b)) do d
        i = findfirst(==(d), inner_dims_b)
        !isnothing(i) && return inner_inds[i]

        i = findfirst(==(d), batch_dims_b)
        !isnothing(i) && return batch_inds[i]

        i = findfirst(==(d), inner_dims_b)
        !isnothing(i) && return outer_inds_b[i]
    end::Vector{Int}

    inds_c = Int[batch_inds; outer_inds_a; outer_inds_b]

    OMEinsum.einsum!((inds_a, inds_b), inds_c, (a, b), c, true, false)
    return c
end

end
