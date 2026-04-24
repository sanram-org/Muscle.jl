module Testing

export construct_test_array

# imported from Reactant.jl
function construct_test_array(::Type{T}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(flat_vector, dims...)
end

function construct_test_array(::Type{Complex{T}}, dims::Int...) where {T<:AbstractFloat}
    flat_vector = collect(T, 1:prod(dims))
    flat_vector ./= prod(dims)
    return reshape(complex.(flat_vector, flat_vector), dims...)
end

function construct_test_array(::Type{T}, dims::Int...) where {T}
    return reshape(collect(T, 1:prod(dims)), dims...)
end

end
