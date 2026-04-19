using Adapt
using ArgCheck

abstract type Platform end

struct PlatformHost <: Platform end
struct PlatformCUDA <: Platform end
struct PlatformReactant <: Platform end
struct PlatformDagger <: Platform end

Platform(::T) where {T<:AbstractArray} = Platform(T)
Platform(::Type{<:Array}) = PlatformHost()
Platform(::Type{T}) where {T<:WrappedArray} = Platform(Adapt.unwrap_type(T))
Platform(x::Tensor) = Platform(parent_type(x))

# TODO promote memspace
function promote_domain(a, b)
    @argcheck Platform(a) == Platform(b) "Platform must be the same"
    return a, b
end

# Base.promote_rule(::Type{PlatformHost}, ::Type{PlatformHost}) = PlatformHost
# Base.promote_rule(::Type{PlatformHost}, ::Type{PlatformCUDA}) = PlatformCUDA
# Base.promote_rule(::Type{PlatformHost}, ::Type{PlatformReactant}) = PlatformReactant

# promote_domain(::A, ::B) where {A<:Platform,B<:Platform} = promote_type(A, B)()

# promote_domain(a, b, c, args...) = promote_domain(promote_domain(a, b), c, args...)
# function promote_domain(a::AbstractArray, b::AbstractArray)
#     target_memspace = promote_domain(memory_space(a), memory_space(b))
#     return adapt_memspace(target_memspace, a), adapt_memspace(target_memspace, b)
# end

# # TODO promote_domain for Tensor

# adapt_memspace(::PlatformHost, x::AbstractArray) = memory_space(x) != PlatformHost() ? adapt(Array, x) : x
# adapt_memspace(::PlatformCUDA, x::AbstractArray) = memory_space(x) != PlatformCUDA() ? adapt(CuArray, x) : x
