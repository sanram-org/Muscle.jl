using Muscle
using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @testset "Index" include("core/index.jl")
    @testset "IndexList" include("core/indexlist.jl")
    @testset "Tensor" include("core/tensor.jl")
    @testset "Operations" verbose = true begin
        @testset "hadamard" include("core/operations/hadamard.jl")
        @testset "unary_einsum" include("core/operations/unary_einsum.jl")
        @testset "binary_einsum" include("core/operations/binary_einsum.jl")
        @testset "tensor_qr" include("core/operations/tensor_qr.jl")
        @testset "tensor_svd" include("core/operations/tensor_svd.jl")
        @testset "tensor_eigen" include("core/operations/tensor_eigen.jl")
        @testset "simple_update" include("core/operations/simple_update.jl")
    end
end

@testset "Integration" verbose = true begin
    @safetestset "OMEinsum" include("integration/omeinsum.jl")
    @safetestset "Strided" include("integration/strided.jl")
    # @safetestset "Dagger" include("integration/dagger.jl")

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
