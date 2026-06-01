function binary_einsum(a::AbstractArray, b::AbstractArray; dims)
    @assert dims isa Tuple && length(dims) == 2
    _platform = promote_platform(platform(a), platform(b))
    backend = getbackend(binary_einsum, _platform)
    return binary_einsum(backend, a, b, dims)
end

Base.@nospecializeinfer function binary_einsum(@nospecialize(B::Backend), @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray), _)
    throw(ArgumentError("`binary_einsum` not implemented or not loaded for backend $B"))
end

function binary_einsum!(c::AbstractArray, a::AbstractArray, b::AbstractArray; dims)
    @assert dims isa Tuple && length(dims) == 2
    out_a = Int[size(a, i) for i in 1:ndims(a) if i ∉ dims[1]]
    out_b = Int[size(b, i) for i in 1:ndims(b) if i ∉ dims[2]]
    @assert size(c) == vcat(out_a, out_b)

    _platform = promote_platform(platform(c), platform(a), platform(b))
    backend = getbackend(binary_einsum!, _platform)
    binary_einsum!(backend, c, a, b)
    return c
end

Base.@nospecializeinfer function binary_einsum!(@nospecialize(B::Backend), @nospecialize(c::AbstractArray), @nospecialize(a::AbstractArray), @nospecialize(b::AbstractArray), @nospecialize(dims))
    @debug "Fallback to generic `binary_einsum!` implementation for backend $B with intermediate copying."
    _c = binary_einsum(B, a, b, dims)
    copyto!(parent(c), parent(_c))
    return c
end

function tensor_qr(A::AbstractArray; dims, inplace=false, kwargs...)
    backend = getbackend(tensor_qr, platform(A))
    return tensor_qr(backend, A; dims, inplace, kwargs...)
end

Base.Base.@nospecializeinfer function tensor_qr(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_qr` not implemented or not loaded for backend $backend"))
end

function tensor_svd(A::AbstractArray; dims, inplace=false, kwargs...)
    backend = getbackend(tensor_svd, platform(A))
    return tensor_svd(backend, A; dims, inplace, kwargs...)
end

Base.@nospecializeinfer function tensor_svd(backend::Backend, @nospecialize(A); kwargs...)
    throw(ArgumentError("`tensor_svd` not implemented or not loaded for backend $backend"))
end

function tensor_eigen(A::AbstractArray; dims, inplace=false, kwargs...)
    backend = getbackend(tensor_eigen, platform(A))
    return tensor_eigen(backend, A; dims, inplace, kwargs...)
end

function tensor_bieigen(A::AbstractArray; dims, inplace=false, kwargs...)
    backend = getbackend(tensor_bieigen, platform(A))
    return tensor_bieigen(backend, A; dims, inplace, kwargs...)
end

# absorb behavior trait
# used to keep type-inference happy (`DontAbsorb` returns 3 tensors, while the rest return 2)
abstract type AbsorbBehavior end
struct DontAbsorb <: AbsorbBehavior end
struct AbsorbU <: AbsorbBehavior end
struct AbsorbV <: AbsorbBehavior end
struct AbsorbEqually <: AbsorbBehavior end

# TODO automatically move to GPU if G are on CPU?
function simple_update(
    A::Tensor,
    ind_physical_a,
    B::Tensor,
    ind_physical_b,
    ind_bond_ab,
    G::Tensor,
    ind_physical_g_a,
    ind_physical_g_b;
    kwargs...,
)
    _platform = promote_platform(platform(A), platform(B), platform(G))
    backend = getbackend(simple_update, _platform)

    return simple_update(
        backend, A, ind_physical_a, B, ind_physical_b, ind_bond_ab, G, ind_physical_g_a, ind_physical_g_b; kwargs...
    )
end

# include("base/hadamard.jl")
# include("unary_einsum.jl")
include("base/binary_einsum.jl")
include("base/tensor_qr.jl")
include("base/tensor_svd.jl")
include("base/tensor_eigen.jl")
include("base/simple_update.jl")
