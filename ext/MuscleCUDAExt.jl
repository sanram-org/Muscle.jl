module MuscleCUDAExt

using Muscle
using Muscle: BackendCuTENSOR
using CUDA
using cuTENSOR
using Base: @nospecializeinfer

function __init__()
    Muscle.register_backend!(BackendCUDA())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum, BackendCuTENSOR())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum!, BackendCuTENSOR())
end

@nospecializeinfer Muscle.platform(@nospecialize(::CuArray)) = Muscle.PlatformCUDA()

## `CUDA` (uses cuTENSOR)
@nospecializeinfer function Muscle.binary_einsum(
    ::BackendCuTENSOR,
    @nospecializeinfer(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    inner_dims_a, inner_dims_b = contracting_dims
    batch_dims_a, batch_dims_b = batching_dims
    outer_dims_a = Int[i for i in 1:ndims(a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(b) if i ∉ inner_dims_b && i ∉ batch_dims_b]
    size_c = vcat(
        Int[size(a, i) for i in batch_dims_a],
        Int[size(a, i) for i in outer_dims_a],
        Int[size(b, i) for i in outer_dims_b],
    )

    T = Base.promote_eltype(a, b)
    c = CUDA.zeros(T, Tuple(size_c))
    binary_einsum!(BackendCuTENSOR(), c, a, b; contracting_dims, batching_dims)
    return c
end

@nospecializeinfer function Muscle.binary_einsum!(
    ::BackendCuTENSOR,
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    inner_dims_a, inner_dims_b = contracting_dims
    batch_dims_a, batch_dims_b = batching_dims
    outer_dims_a = Int[i for i in 1:ndims(a) if i ∉ inner_dims_a && i ∉ batch_dims_a]
    outer_dims_b = Int[i for i in 1:ndims(b) if i ∉ inner_dims_b && i ∉ batch_dims_b]

    modes_a = collect(1:ndims(a))
    modes_b = ndims(a) .+ (1:ndims(b))

    n = ndims(a) + ndims(b) + 1
    for (ai, bi) in zip(inner_dims_a, inner_dims_b)
        modes_a[ai] = modes_b[bi] = n
        n += 1
    end

    for (ai, bi) in zip(inner_dims_a, inner_dims_b)
        modes_a[ai] = modes_b[bi] = n
        n += 1
    end

    modes_c = vcat(
        Int[modes_a[i] for i in batch_dims_a],
        Int[modes_a[i] for i in outer_dims_a],
        Int[modes_b[i] for i in outer_dims_b],
    )

    # call cuTENSOR: op_out(C) := α * opA(A) * opB(B) + β * opC(C)
    α = 1
    β = 0
    op_a = cuTENSOR.OP_IDENTITY
    op_b = cuTENSOR.OP_IDENTITY
    op_c = cuTENSOR.OP_IDENTITY
    op_out = cuTENSOR.OP_IDENTITY
    cuTENSOR.contract!(α, a, modes_a, op_a, b, modes_b, op_b, β, c, modes_c, op_c, op_out)

    return c
end

end
