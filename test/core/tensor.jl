using Test
using Muscle
using Muscle: fuse
using Muscle.Testing
using LinearAlgebra: LinearAlgebra

@testset "Constructors" begin
    @testset "Number" begin
        tensor = Tensor(1.0)
        @test isempty(variance(tensor))
        @test parent(tensor) == fill(1.0)
    end

    @testset "Array" begin
        data = ones(2, 2, 2)
        
        tensor = Tensor(data)
        @test all(==(Invariant), variance(tensor))
        @test parent(tensor) === data

        tensor = Tensor(data, [Covariant, Contravariant, Invariant])
        @test variance(tensor, 1) == Covariant
        @test variance(tensor, 2) == Contravariant
        @test variance(tensor, 3) == Invariant
        @test parent(tensor) === data
    end
end

@testset "eltype - $T" for T in [Bool, Int, Float64, Complex{Float64}]
    tensor = Tensor(zeros(T, 2))
    @test eltype(tensor) == T
end

@testset "elsize - $T" for T in [Bool, Int, Float64, Complex{Float64}]
    tensor = Tensor(zeros(T, 2))
    @test Base.elsize(tensor) == sizeof(T)
end

@testset "==" begin
    a = Tensor(zeros(Int, 2, 2, 2))
    b = Tensor(zeros(Int, 2, 2, 2))
    @test a !== b
    @test a == b

    c = Tensor(zeros(2, 2, 2), [Covariant, Contravariant, Invariant])
    @test a != c
end

@testset "isequal" begin
    tensor = Tensor(zeros(2, 2, 2))
    @test tensor == copy(tensor)
    @test tensor != zeros(size(tensor)...)
    @test zeros(size(tensor)...) != tensor

    @test tensor ∈ [tensor]
    @test copy(tensor) ∈ [tensor]
    @test tensor ∈ [copy(tensor)]
    @test zeros(size(tensor)...) ∉ [tensor]

    @test tensor ∈ Set([tensor])
    @test zeros(size(tensor)...) ∉ Set([tensor])

    @test tensor == permutedims(tensor, (3, 1, 2))
end

@testset "isapprox" begin
    data = rand(2, 3, 4, 5)
    tensor = Tensor(data)

    @test tensor ≈ copy(tensor)
    @test tensor ≈ tensor .+ 1e-14
end

@testset "copy" begin
    tensor = Tensor(zeros(2, 2, 2))
    copied = copy(tensor)
    @test tensor !== copied
    @test tensor == copied

    subtensor = selectdim(tensor, 1, 1)
    copied_subtensor = copy(subtensor)
    @test copied_subtensor isa Tensor{T,N,Array{T,N}} where {T,N}
end

@testset "similar" begin
    tensor = Tensor(ones(2, 2, 2), [Covariant, Contravariant, Invariant])

    @test eltype(similar(tensor)) == eltype(tensor)
    @test size(similar(tensor)) == size(tensor)
    @test parent(similar(tensor)) != parent(tensor)
    @test variance(similar(tensor)) == variance(tensor)

    @test eltype(similar(tensor, Bool)) == Bool
    @test size(similar(tensor, Bool)) == size(tensor)
    @test variance(similar(tensor, Bool)) == variance(tensor)

    @test eltype(similar(tensor, 2, 2, 4)) == eltype(tensor)
    @test size(similar(tensor, 2, 2, 4)) == (2, 2, 4)
    @test variance(similar(tensor, 2, 2, 4)) == variance(tensor)

    @test eltype(similar(tensor, Bool, 2, 2, 4)) == Bool
    @test size(similar(tensor, Bool, 2, 2, 4)) == (2, 2, 4)
    @test variance(similar(tensor, Bool, 2, 2, 4)) == variance(tensor)

    @test_throws DimensionMismatch similar(tensor, 2, 2)
end

@testset "zero" begin
    tensor = Tensor(ones(2, 2, 2), [Covariant, Contravariant, Invariant])
    @test parent(zero(tensor)) == zeros(size(tensor)...)
    @test variance(zero(tensor)) == variance(tensor)
end

@testset "strides" begin
    tensor = Tensor(zeros(2, 2, 2))
    @test strides(tensor) == (1, 2, 4)
    @test stride(tensor, 1) == 1
    @test stride(tensor, 2) == 2
    @test stride(tensor, 3) == 4
end

@testset "unsafe_convert" begin
    tensor = Tensor(zeros(2, 2, 2))
    @test Base.unsafe_convert(Ptr{Float64}, tensor) == Base.unsafe_convert(Ptr{Float64}, parent(tensor))
end

@testset "getindex" begin
    data = [1 2; 3 4]
    tensor = Tensor(data)

    @test tensor[1, 1] == 1
    @test tensor[1, 2] == 2
    @test tensor[2, 1] == 3
    @test tensor[2, 2] == 4

    @test tensor[1 => 1, 2 => 1] == 1
    @test tensor[1 => 1, 2 => 2] == 2
    @test tensor[1 => 2, 2 => 1] == 3
    @test tensor[1 => 2, 2 => 2] == 4

    # partial indexing
    @test tensor[1, :] == [1, 2]
    @test tensor[2, :] == [3, 4]
    @test tensor[:, 1] == [1, 3]
    @test tensor[:, 2] == [2, 4]

    @test tensor[:, :] == data[:, :]
    @test tensor[:] == data[:]

    @test tensor[1 => 1] == [1, 2]
    @test tensor[1 => 2] == [3, 4]
    @test tensor[2 => 1] == [1, 3]
    @test tensor[2 => 2] == [2, 4]

    # 0-dim indexing
    @testset let tensor = Tensor(fill(1))
        @test tensor[] == 1
    end
end

@testset "setindex!" begin
    data = [1 2; 3 4]
    tensor = Tensor(data)

    # assign scalar
    @testset let tensor = copy(tensor)
        tensor[1, 1] = 0
        @test tensor[1, 1] == 0
    end

    @testset let tensor = copy(tensor)
        tensor[1 => 1, 2 => 1] = 0
        @test tensor[1, 1] == 0
    end

    # assign array
    @testset let tensor = copy(tensor)
        tensor[1, :] = [5, 5]
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[1 => 1] = [5, 5]
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[:, 1] = [6, 6]
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[2 => 1] = [6, 6]
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[:, :] = data * 5
        @test tensor[:, :] == data * 5
    end

    @testset let tensor = copy(tensor)
        tensor[:] = data[:] * 10
        @test tensor[:] == data[:] * 10
    end

    # broadcasting assignment
    @testset let tensor = copy(tensor)
        tensor[1, :] .= 5
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[1 => 1] .= 5
        @test tensor[1, :] == [5, 5]
    end

    @testset let tensor = copy(tensor)
        tensor[:, 1] .= 6
        @test tensor[:, 1] == [6, 6]
    end

    @testset let tensor = copy(tensor)
        tensor[2 => 1] .= 6
        @test tensor[:, 1] == [6, 6]
    end

    # 0-dim assignment
    @testset let tensor = Tensor(fill(1))
        tensor[] = 2
        @test tensor[] == 2
    end
end

@testset "other indexing methods" begin
    data = ones(2, 2, 2)
    tensor = Tensor(data)

    @test firstindex(tensor) == 1
    @test lastindex(tensor) == 8
    @test all(firstindex(tensor, i) == 1 for i in 1:ndims(tensor))
    @test all(lastindex(tensor, i) == 2 for i in 1:ndims(tensor))

    @test axes(tensor) == axes(data)
    @test first(tensor) == first(data)
    @test last(tensor) == last(data)

    for i in [0, -1, length(tensor) + 1]
        @test_throws BoundsError tensor[i]
    end
end

@testset "iteration" begin
    data = construct_test_array(Int, 2, 2, 2)
    tensor = Tensor(data)

    @test Base.IteratorSize(tensor) == Base.HasShape{3}()
    @test Base.IteratorEltype(tensor) == Base.HasEltype()
    @test all(x -> ==(x...), zip(tensor, data))
end

@testset "broadcasting" begin
    data = construct_test_array(Int, 2, 2, 2)
    @test begin
        tensor = Tensor(data)
        tensor = tensor .+ one(eltype(tensor))

        parent(tensor) == data .+ one(eltype(tensor))
    end

    @test begin
        tensor = Tensor(data)
        tensor = sin.(tensor)

        parent(tensor) == sin.(data)
    end
end

@testset "selectdim" begin
    data = construct_test_array(Int, 2, 2, 2)
    tensor = Tensor(data, [Covariant, Contravariant, Invariant])

    @test parent(selectdim(tensor, 1, 1)) == selectdim(data, 1, 1)
    @test parent(selectdim(tensor, 2, 2)) == selectdim(data, 2, 2)
    @test variance(selectdim(tensor, 1, 1)) == [Contravariant, Invariant]
    @test variance(selectdim(tensor, 1, 1:1)) == [Covariant, Contravariant, Invariant]
end

@testset "view" begin
    data = construct_test_array(Int, 2, 2, 2)
    tensor = Tensor(data)

    @test parent(view(tensor, 2, :, :)) == view(data, 2, :, :)
    @test parent(view(tensor, 1 => 1)) == view(data, 1, :, :)
    @test parent(view(tensor, 2 => 2)) == view(data, :, 2, :)
    @test parent(view(tensor, 1 => 2, 3 => 1)) == view(data, 2, :, 1)
    @test parent(view(tensor, 1 => 1:1)) == view(data, (1:1), :, :)

    data = [1 2; 3 4]
    tensor = Tensor(data, [Covariant, Contravariant])

    @test @view(tensor[1, 1]) == Tensor(fill(1))
    @test @view(tensor[1, 2]) == Tensor(fill(2))
    @test @view(tensor[2, 1]) == Tensor(fill(3))
    @test @view(tensor[2, 2]) == Tensor(fill(4))

    # partial indexing
    @test @view(tensor[1, :]) == Tensor([1, 2], [Contravariant])
    @test @view(tensor[2, :]) == Tensor([3, 4], [Contravariant])
    @test @view(tensor[:, 1]) == Tensor([1, 3], [Covariant])
    @test @view(tensor[:, 2]) == Tensor([2, 4], [Covariant])

    @test @view(tensor[:, :]) == Tensor(@view(data[:, :]), [Covariant, Contravariant])
    @test @view(tensor[:]) == Tensor(@view(data[:]), [Invariant])

    @test @view(tensor[1 => 1]) == Tensor([1, 2], [Contravariant])
    @test @view(tensor[1 => 2]) == Tensor([3, 4], [Contravariant])
    @test @view(tensor[2 => 1]) == Tensor([1, 3], [Covariant])
    @test @view(tensor[2 => 2]) == Tensor([2, 4], [Covariant])

    # 0-dim indexing
    @testset let tensor = Tensor(fill(1))
        @test @view(tensor[]) == Tensor(@view(parent(tensor)[]))
        @test @view(tensor[])[] == 1
    end
end

@testset "permutedims" begin
    data = reshape(collect(1:24), 2, 3, 4)
    tensor = Tensor(data, [Covariant, Contravariant, Invariant])

    perm = (3, 1, 2)
    c = permutedims(tensor, perm)
    @test variance(c) == [Invariant, Covariant, Contravariant]
    @test parent(c) == permutedims(data, perm)

    newtensor = Tensor(similar(data, 4, 2, 3), [Invariant, Covariant, Contravariant])
    permutedims!(newtensor, tensor, perm)
    @test parent(newtensor) == parent(c)
end

@testset "conj" begin
    @testset "scalar" begin
        a = Tensor(fill(1.0 + 1.0im))
        b = conj(a)

        @test isempty(variance(b))
        @test isapprox(parent(b), 1.0 - 1.0im)
    end

    @testset "Vector" begin
        data = fill(1.0 + 1.0im, 2)
        a = Tensor(data, [Covariant])
        b = conj(a)

        @test variance(b) == [Covariant]
        @test all(isapprox.(b, fill(1.0 - 1.0im, size(data)...)))
    end

    @testset "Matrix" begin
        data = fill(1.0 + 1.0im, 2, 2)
        a = Tensor(data, [Covariant, Contravariant])
        b = conj(a)

        @test variance(b) == [Covariant, Contravariant]
        @test all(isapprox.(b, fill(1.0 - 1.0im, size(data)...)))
    end
end

@testset "adjoint" begin
    @testset "scalar" begin
        a = Tensor(fill(1.0 + 1.0im))
        b = adjoint(a)

        @test isempty(variance(b))
        @test isapprox(parent(b), 1.0 - 1.0im)
    end

    @testset "Vector" begin
        data = fill(1.0 + 1.0im, 2)
        a = Tensor(data, [Covariant])
        b = adjoint(a)

        @test variance(b) == [Contravariant]
        @test all(isapprox.(b, fill(1.0 - 1.0im, size(data)...)))
    end

    @testset "Matrix" begin
        data = fill(1.0 + 1.0im, 2, 2)
        a = Tensor(data, [Covariant, Contravariant])
        b = adjoint(a)

        @test variance(b) == [Contravariant, Covariant]
        @test all(isapprox.(b, fill(1.0 - 1.0im, size(data)...)))
    end
end

@testset "transpose" begin
    @testset "Vector" begin
        data = construct_test_array(Complex{Float64}, 2)
        a = Tensor(data)
        b = transpose(a)

        @test ndims(b) == 1
        @test all(isapprox.(b, data))
    end

    @testset "Matrix" begin
        data = construct_test_array(Complex{Float64}, 2, 2)
        a = Tensor(data, [Covariant, Contravariant])
        b = transpose(a)

        @test variance(b) == [Contravariant, Covariant]
        @test ndims(b) == 2
        @test all(isapprox.(b, transpose(data)))
    end
end

@testset "extend" begin
    data = construct_test_array(Int, 2, 2, 2)
    tensor = Tensor(data, [Covariant, Contravariant, Covariant])

    let new = Muscle.extend(tensor; axis=1)
        @test variance(new) == [Invariant, Covariant, Contravariant, Covariant]
        @test size(new, 1) == 1
        @test selectdim(new, 1, 1) == tensor
    end

    let new = Muscle.extend(tensor; axis=4)
        @test variance(new) == [Covariant, Contravariant, Covariant, Invariant]
        @test size(new, 4) == 1
        @test selectdim(new, 4, 1) == tensor
    end

    let new = Muscle.extend(tensor; axis=1, size=2, method=:zeros)
        @test variance(new) == [Invariant, Covariant, Contravariant, Covariant]
        @test size(new, 1) == 2
        @test selectdim(new, 1, 1) == tensor
        @test selectdim(new, 1, 2) == Tensor(zeros(size(data)...), variance(tensor))
    end

    let new = Muscle.extend(tensor; axis=1, size=2, method=:repeat)
        @test variance(new) == [Invariant, Covariant, Contravariant, Covariant]
        @test size(new, 1) == 2
        @test selectdim(new, 1, 1) == tensor
        @test selectdim(new, 1, 2) == tensor
    end
end

@testset "expand" begin
    data = construct_test_array(Int, 2, 2, 2)
    tensor = Tensor(data, [Covariant, Contravariant, Covariant])

    let new = Muscle.expand(tensor, 1, 3; method=:zeros)
        @test variance(new) == [Covariant, Contravariant, Covariant]
        @test size(new, 1) == 3
        @test view(new, 1 => 1:2) == tensor
        @test view(new, 1 => 3:3) ≈ Tensor(zeros(Tuple(d == 1 ? 1 : size(tensor, d) for d in 1:ndims(tensor))), variance(tensor))
    end

    let new = Muscle.expand(tensor, 2, 3; method=:zeros)
        @test variance(new) == [Covariant, Contravariant, Covariant]
        @test size(new, 2) == 3
        @test view(new, 2 => 1:2) == tensor
        @test view(new, 2 => 3:3) ≈
            Tensor(zeros(Tuple(d == 2 ? 1 : size(tensor, d) for d in 1:ndims(tensor))), variance(tensor))
    end

    let new = Muscle.expand(tensor, 1, 3; method=:rand)
        @test variance(new) == [Covariant, Contravariant, Covariant]
        @test size(new, 1) == 3
        @test view(new, 1 => 1:2) == tensor
    end
end

@testset "Base.cat" begin
    data = construct_test_array(Int, 2, 2, 2)
    tensor = Tensor(data, [Covariant, Contravariant, Covariant])

    @testset let pad = Tensor(zeros(3, 2, 2), [Covariant, Contravariant, Covariant])
        new = cat(tensor, pad; dims=1)
        @test variance(new) == [Covariant, Contravariant, Covariant]
        @test size(new, 1) == 5
        @test view(new, 1 => 1:2) == tensor
        @test view(new, 1 => 3:5) == pad
    end

    @testset let pad = Tensor(zeros(2, 3, 2), [Covariant, Contravariant, Covariant])
        new = cat(tensor, pad; dims=2)
        @test variance(new) == [Covariant, Contravariant, Covariant]
        @test size(new, 2) == 5
        @test view(new, 2 => 1:2) == tensor
        @test view(new, 2 => 3:5) == pad
    end

    @testset let pad = Tensor(zeros(2, 2, 3), [Covariant, Contravariant, Covariant])
        new = cat(tensor, pad; dims=3)
        @test variance(new) == [Covariant, Contravariant, Covariant]
        @test size(new, 3) == 5
        @test view(new, 3 => 1:2) == tensor
        @test view(new, 3 => 3:5) == pad
    end

    # TODO test cat on more than 1 dim
end

@testset "fuse" begin
    tensor = Tensor(construct_test_array(Int, 2, 3), [Covariant, Contravariant])
    @test_throws AssertionError fuse(tensor, (1,2))

    tensor = Tensor(construct_test_array(Int, 2, 3), [Covariant, Covariant])
    grouped = fuse(tensor, [1,2])
    @test vec(tensor) ≈ parent(grouped)

    grouped = fuse(tensor, [2,1])
    @test vec(transpose(parent(tensor))) ≈ parent(grouped)

    tensor = Tensor(construct_test_array(Int, 2, 3, 4), [Covariant, Contravariant, Covariant])
    grouped = fuse(tensor, [1, 3])
    @test reshape(permutedims(parent(tensor), [2, 1, 3]), 3, 8) ≈ parent(grouped)

    grouped = fuse(tensor, [3, 1])
    @test reshape(permutedims(parent(tensor), [2, 3, 1]), 3, 8) ≈ parent(grouped)
end

@testset "sum" begin
    tensor = Tensor(ones(Int, 2, 3, 4), [Covariant, Contravariant, Invariant])

    @test sum(tensor) == Tensor(fill(sum(parent(tensor))))
    @test sum(tensor; dims=1) == Tensor(sum(parent(tensor); dims=1), [Covariant, Contravariant, Invariant])
    @test sum(tensor; dims=2) == Tensor(sum(parent(tensor); dims=2), [Covariant, Contravariant, Invariant])
    @test sum(tensor; dims=3) == Tensor(sum(parent(tensor); dims=3), [Covariant, Contravariant, Invariant])
end

@testset "isisometry" begin
    tensor = Tensor(collect(LinearAlgebra.I(8)))
    @test isisometry(tensor, 1)
    @test isisometry(tensor, 2)

    tensor = Tensor(reshape(collect(LinearAlgebra.I(8)), 2, 4, 8))
    @test !isisometry(tensor, 1)
    @test !isisometry(tensor, 2)
    @test isisometry(tensor, 3)

    tensor = Tensor(ones(8, 8))
    @test !isisometry(tensor, 1)
    @test !isisometry(tensor, 2)
end
