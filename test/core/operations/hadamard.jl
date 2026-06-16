using Test
using Muscle
using Muscle: hadamard, hadamard!

@testset "tensor - scalar bcast" begin
    a = Tensor(ones(2, 3, 4))
    b = Tensor(fill(2.0))

    let c = hadamard(a, b; dims=((),()))
        @test parent(c) ≈ 2.0 .* ones(size(a)...)
    end

    let a = copy(a)
        c = hadamard!(a, b; dims=((),()))
        @test c === a
        @test parent(a) ≈ 2.0 .* ones(size(a)...)
    end
end

@testset "tensor - vector bcast" begin
    a = Tensor(ones(2, 3, 4))

    let b = Tensor([1.0, 2.0])
        let c = hadamard(a, b; dims=([1], [1]))
            @test size(c) == (2, 3, 4)
            for d in size(c, 1)
                @test all(selectdim(c, 1, d) .≈ d)
            end
        end

        let a = copy(a)
            c = hadamard!(a, b; dims=([1], [1]))
            @test c === a
            for d in size(c, 1)
                @test all(selectdim(a, 1, d) .≈ d)
            end
        end
    end

    let b = Tensor([1.0, 2.0, 3.0])
        let c = hadamard(a, b; dims=[[2], [1]])
            @test size(c) == (2, 3, 4)
            for d in size(c, 2)
                @test all(selectdim(c, 2, d) .≈ d)
            end
        end

        let a = copy(a)
            c = hadamard!(a, b; dims=[[2], [1]])
            @test c === a
            for d in size(c, 2)
                @test all(selectdim(a, 2, d) .≈ d)
            end
        end
    end

    let b = Tensor([1.0, 2.0, 3.0, 4.0])
        let c = hadamard(a, b; dims=((3,), (1,)))
            @test size(c) == (2, 3, 4)
            for d in size(c, 3)
                @test all(selectdim(c, 3, d) .≈ d)
            end
        end

        let a = copy(a)
            c = hadamard!(a, b; dims=((3,), (1,)))
            @test c === a
            for d in size(c, 3)
                @test all(selectdim(a, 3, d) .≈ d)
            end
        end
    end
end

@testset "tensor - tensor bcast" begin
    a = Tensor(ones(2, 3, 4))
    b = Tensor(Float64[1 2 3; 4 5 6])

    let c = hadamard(a, b; dims=[[1,2], [1,2]])
        @test size(c) == (2, 3, 4)

        for coord in eachindex(IndexCartesian(), b)
            view_c = view(c, [dim => i for (dim, i) in zip(1:ndims(b), Tuple(coord))]...)
            @test all(view_c .≈ b[coord])
        end
    end

    let a = copy(a)
        c = hadamard!(a, b; dims=[[1,2], [1,2]])
        @test c === a

        for coord in eachindex(IndexCartesian(), b)
            view_a = view(a, [dim => i for (dim, i) in zip(1:ndims(b), Tuple(coord))]...)
            @test all(view_a .≈ b[coord])
        end
    end
end

@testset "tensor - tensor" begin
    a = Tensor(ones(2, 3, 4))
    b = Tensor(2ones(2, 3, 4))

    let c = hadamard(a, b; dims=[[1,2,3], [1,2,3]])
        @test size(c) == (2, 3, 4)
        @test all(c .≈ 2.0)
    end

    let a = copy(a)
        c = hadamard!(a, b; dims=[[1,2,3], [1,2,3]])
        @test c === a
        @test all(a .≈ 2.0)
    end
end
