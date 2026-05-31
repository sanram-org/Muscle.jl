using Test
using Muscle
using Muscle.Testing
using Reactant
using Adapt
using Enzyme
using OMEinsum

# temporal fix
@warn "Setting default backend to CPU for testing."
Reactant.set_default_backend("cpu")

# TODO test `make_tracer`
# TODO test `create_result`
# TODO test `traced_getfield`

# TODO test unary einsum
# TODO test scalar × tensor
@testset "conj" begin
    A = Tensor(rand(ComplexF64, 2, 3), [Index(:i), Index(:j)])
    Are = adapt(ConcreteRArray, A)

    C = conj(A)
    Cre = @jit conj(Are)

    @test Cre ≈ C
end

@testset "hadamard" begin
    # same shape
    A = Tensor(construct_test_array(Float64, 2, 3), [Index(:i), Index(:j)])
    B = Tensor(construct_test_array(Float64, 2, 3), [Index(:i), Index(:j)])
    C = hadamard(A, B)

    Are = adapt(ConcreteRArray, A)
    Bre = adapt(ConcreteRArray, B)
    Cre = @jit hadamard(Are, Bre)
    @test @allowscalar Cre ≈ C

    # propagation through one axis
    D = Tensor(construct_test_array(Float64, 2), [Index(:i)])
    E = hadamard(A, D)

    Dre = adapt(ConcreteRArray, D)
    Ere = @jit hadamard(Are, Dre)
    @test @allowscalar Ere ≈ E

    # propagation through another axis
    F = Tensor(construct_test_array(Float64, 3), [Index(:j)])
    G = hadamard(A, F)

    Fre = adapt(ConcreteRArray, F)
    Gre = @jit hadamard(Are, Fre)
    @test @allowscalar Gre ≈ G
end

@testset "binary_einsum" begin
    @testset "matrix multiplication - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 3), [Index(:i), Index(:j)])
        B = Tensor(rand(T, 3, 4), [Index(:j), Index(:k)])
        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)

        @testset "without permutation" begin
            C = binary_einsum(A, B)
            Cre = @jit binary_einsum(Are, Bre)
            @test Cre ≈ C

            @testset "hybrid" begin
                Cre = @jit binary_einsum(A, Bre)
                @test Cre ≈ C
            end
        end

        @testset "with permutation" begin
            f(a, b) = binary_einsum(a, b; out=Index.([:k, :i]))
            C = f(A, B)
            Cre = @jit f(Are, Bre)
            @test Cre ≈ C

            @testset "hybrid" begin
                Cre = @jit f(A, Bre)
                @test Cre ≈ C
            end
        end
    end

    @testset "inner product - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 3, 4), [Index(:i), Index(:j)])
        B = Tensor(rand(T, 4, 3), [Index(:j), Index(:i)])
        C = binary_einsum(A, B)

        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)
        Cre = @jit binary_einsum(Are, Bre)

        @test @allowscalar Cre ≈ C
    end

    @testset "outer product - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 2), [Index(:i), Index(:j)])
        B = Tensor(rand(T, 2, 2), [Index(:k), Index(:l)])
        C = binary_einsum(A, B)

        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)
        Cre = @jit binary_einsum(Are, Bre)

        @test @allowscalar Cre ≈ C
    end

    @testset "manual - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 3, 4), [Index(:i), Index(:j), Index(:k)])
        B = Tensor(rand(T, 4, 5, 3), [Index(:k), Index(:l), Index(:j)])
        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)

        # binary_einsumion of all common indices
        C = binary_einsum(A, B; dims=[Index(:j), Index(:k)])
        Cre = @jit binary_einsum(Are, Bre; dims=[Index(:j), Index(:k)])

        @test @allowscalar Cre ≈ C

        # binary_einsumion of not all common indices
        # NOTE using `OMEinsum` because we are treating `:k` as a hyperindex
        # TODO better use backend override when available
        C = binary_einsum(Muscle.BackendOMEinsum(), [Index(:i), Index(:k), Index(:l)], A, B)
        Cre = @jit binary_einsum(Are, Bre; dims=[Index(:j)])

        @test @allowscalar Cre ≈ C
    end

    @testset "multiple tensors - eltype=$T" for T in [Float64, ComplexF64]
        A = Tensor(rand(T, 2, 3, 4), [Index(:i), Index(:j), Index(:k)])
        B = Tensor(rand(T, 4, 5, 3), [Index(:k), Index(:l), Index(:j)])
        C = Tensor(rand(T, 5, 6, 2), [Index(:l), Index(:m), Index(:i)])
        D = Tensor(rand(T, 6, 7, 2), [Index(:m), Index(:n), Index(:i)])

        Are = adapt(ConcreteRArray, A)
        Bre = adapt(ConcreteRArray, B)
        Cre = adapt(ConcreteRArray, C)
        Dre = adapt(ConcreteRArray, D)

        f3(a, b, c, d) = binary_einsum(binary_einsum(a, b), binary_einsum(c, d; dims=[Index(:m)]))

        # NOTE using `OMEinsum` because we are treating `:i` as a hyperindex
        # TODO better use backend override when available
        function f3_omeinsum(a, b, c, d)
            binary_einsum(
                binary_einsum(a, b), binary_einsum(Muscle.BackendOMEinsum(), [Index(:l), Index(:i), Index(:n)], c, d)
            )
        end

        X = f3_omeinsum(A, B, C, D)
        Xre = @jit f3(Are, Bre, Cre, Dre)

        @test @allowscalar Xre ≈ X
    end
end

@testset "binary_einsum!" begin
    A = Tensor(rand(2, 3), [Index(:i), Index(:j)])
    B = Tensor(rand(3, 4), [Index(:j), Index(:k)])
    Are = adapt(ConcreteRArray, A)
    Bre = adapt(ConcreteRArray, B)

    C = Tensor(zeros(2, 4), [Index(:i), Index(:k)])
    Cre = adapt(ConcreteRArray, C)

    binary_einsum!(C, A, B)
    @jit binary_einsum!(Cre, Are, Bre)
    @test @allowscalar Cre ≈ C
end

@testset "tensor_svd" begin
    @testset "$T - $Asize" for T in [Float64, ComplexF64], Asize in [(4,4), (4,5), (5,4)]
        A = Tensor(construct_test_array(T, Asize...), [Index(:i), Index(:j)])
        Are = adapt(ConcreteRArray, A)

        ind_s = Index(:x)
        Ure, Sre, Vtre = @jit Muscle.tensor_svd(Are; inds_u=[Index(:i)], ind_s, algorithm="QRIteration");
        Areconstructed = @jit binary_einsum(hadamard(Ure, Sre), Vtre)

        @test @allowscalar isapprox(Areconstructed, A)
    end

    @testset "n-dim setting - $T - case $i" for T in [Float64, ComplexF64],
        (i, inds_u) in enumerate([[Index(:i), Index(:j)], [Index(:k)], [Index(:i)]])

        A = Tensor(construct_test_array(T, 2, 4, 6, 8), [Index(:i), Index(:j), Index(:k), Index(:l)])
        Are = adapt(ConcreteRArray, A)

        ind_s = Index(:x)
        Ure, Sre, Vtre = @jit Muscle.tensor_svd(Are; inds_u, ind_s, algorithm="QRIteration")
        Areconstructed = @jit binary_einsum(hadamard(Ure, Sre), Vtre)

        @test @allowscalar isapprox(Areconstructed, A)
    end
end

@testset "autodiff" begin
    @testset "inner product" begin
        # inner-product of vectors
        @testset let
            A = Tensor([1.0, 2.0], [Index(:i)])
            B = Tensor([3.0, 4.0], [Index(:i)])
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            grad_f(a, b) = Enzyme.gradient(Reverse, binary_einsum, a, b)
            dAre, dBre = @jit grad_f(Are, Bre)

            @test @allowscalar dAre ≈ B
            @test @allowscalar dBre ≈ A
        end

        # inner-product of matrices
        @testset let
            A = Tensor([1.0 2.0; 3.0 4.0], [Index(:i), Index(:k)])
            B = Tensor([5.0 6.0; 7.0 8.0], [Index(:i), Index(:k)])
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            grad_f2(a, b) = Enzyme.gradient(Reverse, binary_einsum, a, b)
            dAre, dBre = @jit grad_f2(Are, Bre)

            @test @allowscalar dAre ≈ B
            @test @allowscalar dBre ≈ A
        end

        # inner-product of complex matrices
        @testset let
            A = Tensor(ComplexF64[1.0 2.0; 3.0im 4.0], [Index(:i), Index(:k)])
            B = Tensor(ComplexF64[5.0 6.0im; 7.0 8.0], [Index(:i), Index(:k)])
            Are = adapt(ConcreteRArray, A)
            Bre = adapt(ConcreteRArray, B)

            grad_f3(a, b) = Enzyme.gradient(Reverse, binary_einsum, a, b)
            dAre, dBre = @jit grad_f3(Are, Bre)

            @test @allowscalar dAre ≈ conj(B)
            @test @allowscalar dBre ≈ conj(A)
        end
    end

    # TODO test other einsum cases
end
