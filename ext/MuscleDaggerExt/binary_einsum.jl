using Muscle
using Dagger: Dagger, ArrayOp, Context, ArrayDomain, EagerThunk, DArray
using LinearAlgebra
using Base: @nospecializeinfer

@nospecializeinfer function Muscle.binary_einsum(::Muscle.BackendDagger, @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray))
    op = BinaryEinsum(a, b, contracting_dims, batching_dims)
    return Dagger._to_darray(op)
end

struct BinaryEinsum{T,N} <: ArrayOp{T,N}
    a::ArrayOp
    b::ArrayOp
    contracting_dims
    batching_dims

    function BinaryEinsum(a, b, contracting_dims, batching_dims)
        inner_dims_a, inner_dims_b = contracting_dims
        _, batch_dims_b = batching_dims

        T = Base.promote_eltype(a, b)
        N = ndims(a) - length(inner_dims_a) + ndims(b) - length(inner_dims_b) - length(batch_dims_b)
        da = Dagger._to_darray(a)
        db = Dagger._to_darray(b)
        return new{T, N}(da, db, contracting_dims, batching_dims)
    end
end

function Base.size(@nospecialize(x::BinaryEinsum))
    inner_dims_a, inner_dims_b = x.contracting_dims
    batch_dims_a, batch_dims_b = x.batching_dims

    return Tuple(vcat(
        Int[size(x.a, i) for i in batch_dims_a],
        Int[size(x.a, i) for i in 1:ndims(x.a) if i ∉ inner_dims_a && i ∉ batch_dims_a],
        Int[size(x.b, i) for i in 1:ndims(x.b) if i ∉ inner_dims_b && i ∉ batch_dims_b],
    ))
end

function Dagger.Blocks(@nospecialize(x::BinaryEinsum))
    inner_dims_a, inner_dims_b = x.contracting_dims
    batch_dims_a, batch_dims_b = x.batching_dims

    return Dagger.Blocks(vcat(
        Int[x.a.partitioning.blocksize[i] for i in batch_dims_a],
        Int[x.a.partitioning.blocksize[i] for i in 1:ndims(x.a) if i ∉ inner_dims_a && i ∉ batch_dims_a],
        Int[x.b.partitioning.blocksize[i] for i in 1:ndims(x.b) if i ∉ inner_dims_b && i ∉ batch_dims_b],
    ))
end

function Dagger.stage(::Context, op::BinaryEinsum{T,N}) where {T,N}
    domain = Dagger.ArrayDomain([1:l for l in size(op)])
    partitioning = Dagger.Blocks(op)

    # NOTE careful with ÷ for dividing into partitions
    subdomains = Array{ArrayDomain{N,NTuple{2,UnitRange{Int}}}}(undef, map(÷, size(op), partitioning.blocksize))
    for indices in eachindex(IndexCartesian(), subdomains)
        subdomains[indices] = ArrayDomain(
            map(Tuple(indices), partitioning.blocksize) do i, step
                (i - 1) * step .+ (1:step)
            end,
        )
    end

    inner_dims_a, inner_dims_b = op.contracting_dims
    batch_dims_a, batch_dims_b = op.batching_dims
    outer_dims_a = Int[i for i in 1:ndims(x.a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(x.b) if i ∉ inner_dims_b && i ∉ batch_dims_b]

    mask_a = vcat(trues(length(outer_dims_a)), falses(length(outer_dims_b)))
    mask_b = .!mask_a

    chunk_perm_a = invperm(vcat(batch_dims_a, outer_inds_a, inner_inds_a))
    chunk_perm_b = invperm(vcat(batch_dims_b, outer_inds_b, inner_inds_b))

    chunks = similar(subdomains, EagerThunk)
    for indices in eachindex(IndexCartesian(), chunks)
        outer_chunk_inds_a = Tuple(indices)[mask_a]
        outer_chunk_inds_b = Tuple(indices)[mask_b]
        
        chunks[indices] = Dagger.treereduce(
            Dagger.AddComputeOp,
            map(Iterators.product([1:size(Dagger.chunks(op.a), i) for i in inner_dims_a]...)) do inner_chunk_inds_a
                chunk_ind_a = permute!(vcat(outer_chunk_inds_a, inner_chunk_inds_a), chunk_perm_a)
                chunk_ind_b = permute!(vcat(outer_chunk_inds_b, inner_chunk_inds_b), chunk_perm_b)

                chunk_a = getindex(Dagger.chunks(op.a), chunk_ind_a...)
                chunk_b = getindex(Dagger.chunks(op.b), chunk_ind_b...)

                # TODO add ThunkOptions: alloc_util, occupancy, ...
                Dagger.@spawn Muscle.binary_einsum(chunk_a, chunk_b; contracting_dims, batching_dims)
            end,
        )
    end

    return DArray(T, domain, subdomains, chunks, partitioning)
end
