using Test
using Muscle
using Muscle: binary_einsum
using CUDA
using CUDA: @allowscalar
using cuTENSOR

@testset "matmul: ij,jk->ik" begin
    a = CUDA.ones(2, 3)
    b = CUDA.ones(3, 4)
    c = binary_einsum(a, b; contracting_dims=[[2], [1]])
    @test @allowscalar c == 3 * CUDA.ones(2, 4)
end

@testset "inner product: ij,ji->" begin
    a = CUDA.ones(3, 4)
    b = CUDA.ones(4, 3)
    c = binary_einsum(a, b; contracting_dims=[[2, 1], [1, 2]])
    @test @allowscalar c == fill(12)
end

@testset "outer product: ij,kl->ijkl" begin
    a = CUDA.ones(2, 3)
    b = CUDA.ones(4, 5)
    c = binary_einsum(a, b; contracting_dims=[Int[], Int[]])
    @test @allowscalar c == fill(1, 2, 3, 4, 5)
end

@testset "scale: ij,->ij" begin
    a = CUDA.ones(2, 3)
    α = CuArray(fill(2.0))

    let c = binary_einsum(a, α; contracting_dims=((), ()))
        @test @allowscalar c == α[] .* a
    end

    let c = binary_einsum(α, a; contracting_dims=((), ()))
        @test @allowscalar c == α[] .* a
    end
end

# hyperindices not yet supported on backendbase
@testset "batch matmul: ijb,jkb->ikb" begin
    a = CUDA.ones(2, 3, 6)
    b = CUDA.ones(3, 4, 6)

    # TODO there's an numerical error
    @test_throws AssertionError binary_einsum(a, b; contracting_dims=[[2], [1]], batching_dims=[[3], [3]])
end

@testset "manual" begin
    @testset "eltype = $T" for T in [Float64, ComplexF64]
        a = CUDA.ones(T, 2, 3, 4)
        b = CUDA.ones(T, 4, 5, 3)

        # contraction of all common indices
        @testset "ijk,klj->il" begin
            c = binary_einsum(a, b; contracting_dims=[[2, 3], [3, 1]])
            @test c ≈ begin
                a_mat = reshape(a, 2, 12)
                b_mat = reshape(permutedims(b, [3, 1, 2]), 12, 5)
                a_mat * b_mat
            end
        end

        # contraction of not all common indices
        # hyperindices not supported on backendbase
        @testset "ijk,klj->ikl" begin
            @test_throws AssertionError binary_einsum(a, b; contracting_dims=[[2], [3]], batching_dims=[[3], [1]])
        end
    end
end
