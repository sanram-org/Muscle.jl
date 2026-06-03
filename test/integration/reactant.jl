using Test
using Muscle
using Muscle: binary_einsum
using Muscle.Testing
using Reactant
using Adapt
using Enzyme

# temporal fix
@warn "Setting default backend to CPU for testing."
Reactant.set_default_backend("cpu")

# TODO test `make_tracer`
# TODO test `create_result`
# TODO test `traced_getfield`

# TODO test unary einsum
# TODO test scalar × tensor
@testset "conj" begin
    A = construct_test_array(ComplexF64, 2, 3)
    Are = adapt(ConcreteRArray, A)

    C = conj(A)
    Cre = @jit conj(Are)

    @test Cre ≈ C
end

@testset "binary_einsum" begin
    @testset "matmul: ij,jk->ik" begin
        a = Reactant.to_rarray(ones(2, 3))
        b = Reactant.to_rarray(ones(3, 4))
        c = @jit binary_einsum(a, b; contracting_dims=[[2], [1]])
        @test c == 3 * ones(2, 4)
    end

    @testset "inner product: ij,ji->" begin
        a = Reactant.to_rarray(ones(3, 4))
        b = Reactant.to_rarray(ones(4, 3))
        c = @jit binary_einsum(a, b; contracting_dims=[[2, 1], [1, 2]])
        @test c == fill(12)
    end

    @testset "outer product: ij,kl->ijkl" begin
        a = Reactant.to_rarray(ones(2, 3))
        b = Reactant.to_rarray(ones(4, 5))
        c = @jit binary_einsum(a, b; contracting_dims=[Int[], Int[]])
        @test c == fill(1, 2, 3, 4, 5)
    end

    @testset "scale: ij,->ij" begin
        a = Reactant.to_rarray(ones(2, 3))
        α = Reactant.to_rarray(fill(2.0))

        let c = @jit binary_einsum(a, α; contracting_dims=((), ()))
            @test c == α[] .* a
        end

        let c = @jit binary_einsum(α, a; contracting_dims=((), ()))
            @test c == α[] .* a
        end
    end

    # hyperindices not yet supported on backendbase
    @testset "batch matmul: ijb,jkb->ikb" begin
        a = Reactant.to_rarray(ones(2, 3, 6))
        b = Reactant.to_rarray(ones(3, 4, 6))

        @test_throws AssertionError @jit binary_einsum(a, b; contracting_dims=[[2], [1]], batching_dims=[[3], [3]])
    end

    @testset "manual" begin
        @testset "eltype = $T" for T in [Float64, ComplexF64]
            a = Reactant.to_rarray(ones(T, 2, 3, 4))
            b = Reactant.to_rarray(ones(T, 4, 5, 3))

            # contraction of all common indices
            @testset "ijk,klj->il" begin
                c = @jit binary_einsum(a, b; contracting_dims=[[2, 3], [3, 1]])
                @test c ≈ begin
                    a_mat = reshape(Array(a), 2, 12)
                    b_mat = reshape(permutedims(Array(b), [3, 1, 2]), 12, 5)
                    a_mat * b_mat
                end
            end

            # contraction of not all common indices
            # hyperindices not supported on backendbase
            @testset "ijk,klj->ikl" begin
                @test_throws AssertionError @jit binary_einsum(
                    a, b; contracting_dims=[[2], [3]], batching_dims=[[3], [1]]
                )
            end
        end
    end
end

@testset "binary_einsum!" begin
    A = construct_test_array(Int, 2, 3)
    B = construct_test_array(Int, 3, 4)
    Are = adapt(ConcreteRArray, A)
    Bre = adapt(ConcreteRArray, B)

    C = zeros(2, 4)
    Cre = adapt(ConcreteRArray, C)

    @jit binary_einsum!(C, A, B)
    @jit binary_einsum!(Cre, Are, Bre)
    @test @allowscalar Cre ≈ C
end

@testset "tensor_svd" begin
    @testset "$T - $Asize" for T in [Float64, ComplexF64], Asize in [(4, 4), (4, 5), (5, 4)]
        A = construct_test_array(T, Asize...)
        Are = adapt(ConcreteRArray, A)

        Ure, Sre, Vtre = @jit Muscle.tensor_svd(Are; dims=[[1], [2]])

        Sre = reshape(Sre, 1, 1, length(Sre))
        Ure = @jit Ure .* Sre
        Areconstructed = @jit binary_einsum(Ure, Vtre; contracting_dims=[[3], [1]])

        @test @allowscalar isapprox(Areconstructed, A)
    end

    @testset "n-dim setting - $T - case $i" for T in [Float64, ComplexF64],
        (i, dims) in enumerate([([1, 2], [3, 4]), ([3, 1], [4, 2])])

        A = construct_test_array(T, 2, 4, 6, 8)
        Are = adapt(ConcreteRArray, A)

        Ure, Sre, Vtre = @jit Muscle.tensor_svd(Are; dims)

        Sre = reshape(Sre, 1, 1, length(Sre))
        Ure = @jit Ure .* Sre
        Areconstructed = @jit binary_einsum(Ure, Vtre; contracting_dims=[[3], [1]])
        perm = Int[dims[1]; dims[2]]
        Areconstructed = @jit permutedims(Areconstructed, perm)

        @test @allowscalar isapprox(Areconstructed, A)
    end
end

@testset "autodiff" begin
    @testset "inner product" begin
        # inner-product of vectors
        @testset let
            A = [1.0, 2.0]
            B = [3.0, 4.0]
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            f(a, b) = binary_einsum(a, b; contracting_dims=[[1], [1]])
            grad_f(a, b) = Enzyme.gradient(Reverse, f, a, b)
            dAre, dBre = @jit grad_f(Are, Bre)

            @test @allowscalar dAre ≈ B
            @test @allowscalar dBre ≈ A
        end

        # inner-product of matrices
        @testset let
            A = [1.0 2.0; 3.0 4.0]
            B = [5.0 6.0; 7.0 8.0]
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            f2(a, b) = binary_einsum(a, b; contracting_dims=[[1, 2], [1, 2]])
            grad_f2(a, b) = Enzyme.gradient(Reverse, f2, a, b)
            dAre, dBre = @jit grad_f2(Are, Bre)

            @test @allowscalar dAre ≈ B
            @test @allowscalar dBre ≈ A
        end

        # inner-product of complex matrices
        @testset let
            A = [1.0 2.0; 3.0im 4.0]
            B = [5.0 6.0im; 7.0 8.0]
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            f3(a, b) = binary_einsum(a, b; contracting_dims=[[1, 2], [1, 2]])
            grad_f3(a, b) = Enzyme.gradient(Reverse, f3, a, b)
            dAre, dBre = @jit grad_f3(Are, Bre)

            @test @allowscalar dAre ≈ conj(B)
            @test @allowscalar dBre ≈ conj(A)
        end
    end

    # TODO test other einsum cases
end
