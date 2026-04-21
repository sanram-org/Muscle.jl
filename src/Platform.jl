using ArgCheck

abstract type Platform end

struct PlatformHost <: Platform end
struct PlatformCUDA <: Platform end
struct PlatformReactant <: Platform end
struct PlatformDagger <: Platform end

platform(x::AbstractArray) = platform(parent(x))
platform(x::Tensor) = platform(parent(x))
platform(::Array) = PlatformHost()

function promote_platform(a::Platform, b::Platform)
    ab = promote_platform_rule(a, b)
    ba = promote_platform_rule(b, a)
    res = promote_platform_result(ab, ba)
    if ismissing(res)
        throw(ArgumentError("No promotion found for $a and $b"))
    else
        return res
    end
end

promote_platform(a, b, c, args...) = promote_platform(promote_platform(a, b), c, args...)
promote_platform(a::AbstractArray, b::AbstractArray) = promote_platform(Platform(a), Platform(b))

promote_platform_result(::Missing, ::Missing) = missing
promote_platform_result(@nospecialize(ab::Platform), ::Missing) = ab
promote_platform_result(::Missing, @nospecialize(ba::Platform)) = ba
promote_platform_result(@nospecialize(ab::Platform), @nospecialize(ba::Platform)) = (@assert(ab == ba); ab)

promote_platform_rule(@nospecialize(::Platform), @nospecialize(::Platform)) = missing
promote_platform_rule(::P, ::P) where {P <: Platform} = P()
promote_platform_rule(::PlatformHost, ::PlatformCUDA) = PlatformCUDA()
promote_platform_rule(::PlatformHost, ::PlatformReactant) = PlatformReactant()
promote_platform_rule(::PlatformHost, ::PlatformDagger) = PlatformDagger()
promote_platform_rule(::PlatformCUDA, ::PlatformReactant) = PlatformReactant()
