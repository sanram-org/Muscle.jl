module MuscleVectorInterfaceExt

using Muscle
using VectorInterface

VectorInterface.scalartype(::Tensor{T}) where {T} = T
VectorInterface.zerovector(t::Tensor, S) = Tensor(zerovector(parent(t), S), variance(t))
VectorInterface.zerovector!(t::Tensor) = t .= zero(eltype(t))
VectorInterface.scale(x::Tensor, α) = Tensor(scale(parent(x), α), variance(x))
VectorInterface.scale!(x::Tensor, α) = x .*= α
VectorInterface.scale!(y::Tensor, x::Tensor, α) = y .= α .* x
VectorInterface.add(y::Tensor, x::Tensor) = y + x
VectorInterface.add(y::Tensor, x::Tensor, α::Number) = y + α * x
VectorInterface.add(y::Tensor, x::Tensor, α::Number, β::Number) = y * β + α * x
VectorInterface.add!(y::Tensor, x::Tensor) = y .= y + x
VectorInterface.add!(y::Tensor, x::Tensor, α::Number) = y .= y + α * x
VectorInterface.add!(y::Tensor, x::Tensor, α::Number, β::Number) = y .= y * β + α * x

VectorInterface.inner(x::Tensor, y::Tensor) = binary_einsum(x, y; contracting_dims=[collect(1:ndims(x)), collect(1:ndims(y))])

end
