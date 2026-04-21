module MuscleAdaptExt

using Muscle
using Adapt

Adapt.adapt_structure(to, x::Tensor) = Tensor(adapt(to, parent(x)), inds(x))

Adapt.parent_type(::Type{Tensor{T,N,A}}) where {T,N,A} = A
Adapt.parent_type(::Type{Tensor{T,N}}) where {T,N} = AbstractArray{T,N}
Adapt.parent_type(::Type{Tensor{T}}) where {T} = AbstractArray{T}
Adapt.parent_type(::Type{Tensor}) = AbstractArray
Adapt.parent_type(::T) where {T<:Tensor} = Adapt.parent_type(T)

end
