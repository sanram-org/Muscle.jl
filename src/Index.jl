"""
    Label{T}

Represents the label of an [`Index`](@ref); i.e. the matching name without variance information.
"""
struct Label{T}
    label::T

    Label(label::T) where {T} = new{T}(label)
end

Label(label::Label) = label
Label(name::String) = Label(Symbol(name))

Base.show(io::IO, label::Label) = print(io, label.label)

"""
    Variance

Variance type of an index. It can be `Covariant`, `Contravariant` or `Invariant`.
"""
@enum Variance begin
    Covariant
    Contravariant
    Invariant # NOTE do not use in principle
end

"""
    Index{T}

Represents an index of a tensor. It consists of a [`Label`](@ref) and [`Variance`](@ref) information.
"""
Base.@kwdef struct Index{T}
    label::Label{T}
    variance::Variance = Invariant

    Index(label::T, variance::Variance = Invariant) where {T} = new{T}(Label(label), variance)
    Index(label::Label{T}, variance::Variance = Invariant) where {T} = new{T}(label, variance)
end

Index(ind::Index, variance::Variance = variance(ind)) = Index(label(ind), variance)

# TODO checkout whether this is a good idea
Base.copy(x::Index) = x

label(ind::Index) = ind.label
variance(ind::Index) = ind.variance

variate(ind::Index, var::Variance) = Index(label(ind), var)

is_equal_label(ind::Index) = Base.Fix1(is_equal_label, ind)
is_equal_label(ind1::Index, ind2::Index) = label(ind1) == label(ind2)

function Base.show(io::IO, ind::Index)
    print(io, "index<")
    print(io, label(ind))
    if variance(ind) == Contravariant
        print(io, "↑")
    elseif variance(ind) == Covariant
        print(io, "↓")
    end
    print(io, ">")
end
