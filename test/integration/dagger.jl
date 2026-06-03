using Test
using Muscle: binary_einsum
using Dagger
using Distributed

procs = addprocs(1)
@everywhere procs using Dagger
@everywhere procs using Muscle

@testset "binary_einsum" begin
    @testset "block-block" begin
        a = Float64[1.0 2.0; 3.0 4.0]
        b = Float64[5.0 6.0; 7.0 8.0]

        block_a = distribute(a, Dagger.Blocks(1, 1))
        block_b = distribute(b, Dagger.Blocks(1, 1))

        c = binary_einsum(a, b; contracting_dims=[[2], [1]])
        block_c = binary_einsum(block_a, block_b; contracting_dims=[[2], [1]])

        @test block_c isa DArray
        @test all(==((1, 1)) ∘ size, Dagger.domainchunks(block_c))
        @test collect(block_c) ≈ c
    end
end

# TODO test with other array types
rmprocs(procs)
