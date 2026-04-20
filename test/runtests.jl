using Muscle
using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @testset "Tensor" include("core/tensor.jl")
    @testset "Operations" verbose = true begin
        @testset "hadamard" include("core/operations/hadamard.jl")
        @testset "unary_einsum" include("core/operations/unary_einsum.jl")
        @testset "binary_einsum" include("core/operations/binary_einsum.jl")
        @testset "tensor_qr_thin" include("core/operations/tensor_qr_thin.jl")
        @testset "tensor_svd_thin" include("core/operations/tensor_svd_thin.jl")
        @testset "tensor_svd_trunc" include("core/operations/tensor_svd_trunc.jl")
        @testset "tensor_eigen_thin" include("core/operations/tensor_eigen_thin.jl")
        @testset "simple_update" include("core/operations/simple_update.jl")
    end
end

@testset "Integration" verbose = true begin
    @safetestset "OMEinsum" include("integration/omeinsum.jl")
    @safetestset "Strided" include("integration/strided.jl")
    @safetestset "Dagger" include("integration/dagger.jl")

    # @safetestset "ChainRules" include("integration/chainrules.jl")

    @safetestset "ITensors" include("integration/itensors.jl")

    if !isnothing(get(ENV, "MUSCLE_TEST_CUDA", nothing))
        @safetestset "CUDA" include("integration/cuda.jl")
    end

    if !isnothing(get(ENV, "MUSCLE_TEST_REACTANT", nothing))
        @safetestset "Reactant" include("integration/reactant.jl")
    end
end

# using Aqua
# Aqua.test_all(Muscle)
