module MuscleReactantExt

using Muscle
using Muscle: BackendReactant
using Reactant
using Reactant:
    @opcall,
    use_overlayed_version,
    TracedRNumber,
    TracedRArray,
    ConcreteRNumber,
    ConcreteRArray,
    AnyTracedRArray,
    AnyConcreteRArray
using Reactant.TracedUtils: set_mlir_data!, get_mlir_data
using PrecompileTools
using LinearAlgebra
using Base: @nospecializeinfer

function __init__()
    Muscle.register_backend!(BackendReactant())

    for op in [Muscle.binary_einsum, Muscle.binary_einsum!, Muscle.tensor_svd]
        Muscle.Operations.register_backend_for_op!(op, BackendReactant())
    end

    muscle_skip_rewrites()
end

# This function is used to skip rewriting of certain functions and type constructors in Reactant.jl, which is necessary
# for overlaying methods called dynamically. By skipping the rewrite where we know it's ok, Julia compilation should 
# take less time. It must be called on the top level for precompilation and in `__init__` for runtime.
function muscle_skip_rewrites()
    Reactant.@skip_rewrite_func Muscle.binary_einsum
    Reactant.@skip_rewrite_func Muscle.nonunique
    Reactant.@skip_rewrite_type Type{<:Muscle.Tensor}
end

for T in [TracedRNumber, ConcreteRNumber, TracedRArray, ConcreteRArray, AnyTracedRArray, AnyConcreteRArray]
    @eval @nospecializeinfer Muscle.platform(@nospecialize(_::$T)) = Muscle.PlatformReactant()
end

# we specify `mode` and `track_numbers` types due to ambiguity
@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
)
    return Tensor
end

@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T}
    return Tensor{TracedRNumber{T}}
end

@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T,N}
    return Tensor{TracedRNumber{T},N}
end

@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(_::Type{Tensor{T,N,A}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    sharding,
    runtime,
) where {T,N,A}
    A_traced = Reactant.traced_type_inner(A, seen, mode, track_numbers, sharding, runtime)
    T_traced = eltype(A_traced)
    return Tensor{T_traced,N,A_traced}
end

@nospecializeinfer function Muscle.binary_einsum(
    ::BackendReactant, @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray); contracting_dims, batching_dims
)
    if !use_overlayed_version(a)
        a = @opcall constant(a)
    end

    if !use_overlayed_version(b)
        b = @opcall constant(b)
    end

    # StableHLO expects matching element types
    T = Base.promote_eltype(a, b)
    da = T.(Reactant.materialize_traced_array(parent(a)))
    db = T.(Reactant.materialize_traced_array(parent(b)))

    contracting_dimensions = collect.(contracting_dims)
    batching_dimensions = collect.(batching_dims)
    c = @opcall dot_general(da, db; contracting_dimensions, batching_dimensions)
    return c
end

@nospecializeinfer function Muscle.binary_einsum!(
    ::BackendReactant,
    @nospecialize(c::AbstractArray),
    @nospecialize(a::AbstractArray),
    @nospecialize(b::AbstractArray);
    contracting_dims,
    batching_dims,
)
    _c = Muscle.binary_einsum(BackendReactant(), a, b; contracting_dims, batching_dims)
    set_mlir_data!(c, get_mlir_data(_c))
    return c
end

# TODO batching dimensions?
@nospecializeinfer function Muscle.tensor_svd(::BackendReactant, @nospecialize(a::AbstractArray); dims, kwargs...)
    inds_u, inds_v = dims

    # permute array
    left_sizes = Int[size(a, i) for i in inds_u]
    right_sizes = Int[size(a, i) for i in inds_v]
    a_mat = permutedims(a, [inds_u; inds_v])
    a_mat = reshape(parent(a_mat), prod(left_sizes), prod(right_sizes))

    a_mat = Reactant.materialize_traced_array(a_mat)

    # error on `cusolver_gesvd`: The GPU implementation of gesvd requires that the input matrix be m x n with m >= n
    # TODO update once fixed in Enzyme-JAX
    apply_fix_for_cusolver_gesvd = get(kwargs, :algorithm, "DEFAULT") == "QRIteration" && size(data, 1) < size(data, 2)

    if apply_fix_for_cusolver_gesvd
        data = @opcall transpose(data, [2, 1]) # modify if batching dims
    end

    U, s, Vt, _ = @opcall svd(data; full=false, kwargs...)

    if apply_fix_for_cusolver_gesvd
        U, Vt = @opcall(transpose(Vt, [2, 1])), @opcall(transpose(U, [2, 1])) # modify if batching dims
    end

    # tensorify results
    U = @opcall reshape(U, left_sizes..., size(s)...)
    Vt = @opcall reshape(Vt, size(s)..., right_sizes...)

    return U, s, Vt
end

# @static if Reactant.Reactant_jll.is_available() && Reactant.precompilation_supported()
#     @setup_workload begin
#         muscle_skip_rewrites()

#         # Initialize the MLIR dialects and set up the XLA client
#         # NOTE taken from https://github.com/EnzymeAD/Reactant.jl/blob/77a9c694c4004cf08b270d08f8a5f51b7bdbf97e/src/Precompile.jl#L57-L83
#         Reactant.initialize_dialect()
#         if Reactant.XLA.REACTANT_XLA_RUNTIME == "PJRT"
#             client = Reactant.XLA.PJRT.CPUClient(; checkcount=false)
#         elseif Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
#             client = Reactant.XLA.IFRT.CPUClient(; checkcount=false)
#         else
#             error("Unsupported runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)")
#         end

#         @compile_workload begin
#             for (Ta, Tb) in [
#                 (Float32, Float32),
#                 (Float64, Float64),
#                 (ComplexF32, ComplexF32),
#                 (ComplexF64, ComplexF64),
#                 (Float64, ComplexF64),
#                 (ComplexF64, Float64),
#             ]
#                 a = Tensor(Reactant.to_rarray(ones(Ta, 2, 2); client), [:i, :j])
#                 b = Tensor(Reactant.to_rarray(ones(Tb, 2, 2); client), [:j, :k])
#                 Reactant.compile(Muscle.einsum, (a, b); client, optimize=:all)
#             end
#         end

#         # clean deinitialization
#         Reactant.XLA.free_client(client)
#         client.client = C_NULL
#         Reactant.deinitialize_dialect()
#         Reactant.clear_oc_cache()
#     end
# end

end
