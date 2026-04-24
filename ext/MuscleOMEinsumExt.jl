module MuscleOMEinsumExt

using Muscle
using Muscle: BackendOMEinsum
using OMEinsum
using ArgCheck

function __init__()
    Muscle.register_backend!(BackendOMEinsum())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.unary_einsum, BackendOMEinsum())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.unary_einsum!, BackendOMEinsum())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum, BackendOMEinsum())
    Muscle.Operations.register_backend_for_op!(Muscle.Operations.binary_einsum!, BackendOMEinsum())
end

function Muscle.unary_einsum(::BackendOMEinsum, inds_y, x::Tensor)
    y = Tensor(similar(parent(x), Tuple(size(x, ind) for ind in inds_y)), inds_y)
    unary_einsum!(BackendOMEinsum(), y, x)
    return y
end

function Muscle.unary_einsum!(::BackendOMEinsum, y::Tensor, x::Tensor)
    @argcheck inds(y) ⊆ inds(x) "Output indices must be a subset of input indices"

    size_dict = Dict(inds(x) .=> size(x))
    OMEinsum.einsum!((inds(x),), inds(y), (parent(x),), parent(y), true, false, size_dict)

    return y
end

function Muscle.binary_einsum(::Muscle.BackendOMEinsum, inds_c, a::Tensor, b::Tensor)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds(a) .=> size(a), inds(b) .=> size(b)])
        size_dict[ind] = ind_size
    end

    data_c = OMEinsum.get_output_array((a, b), Int[size_dict[i] for i in inds_c], false)
    OMEinsum.einsum!((inds(a), inds(b)), inds_c, (parent(a), parent(b)), data_c, true, false, size_dict)
    return Tensor(data_c, inds_c)
end

function Muscle.binary_einsum!(::Muscle.BackendOMEinsum, c::Tensor, a::Tensor, b::Tensor)
    size_dict = Dict{Index,Int}()
    for (ind, ind_size) in Iterators.flatten([inds(a) .=> size(a), inds(b) .=> size(b)])
        size_dict[ind] = ind_size
    end

    OMEinsum.einsum!((inds(a), inds(b)), inds(c), (parent(a), parent(b)), parent(c), true, false, size_dict)
    return c
end

end
