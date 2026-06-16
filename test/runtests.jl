using Muscle
using Test
using SafeTestsets

target_testsets = isempty(ARGS) ? ["core", "integration"] : ARGS

if "core" in target_testsets
    @testset "Unit" verbose = true begin
        @testset "Variance" include("core/variance.jl")
        @testset "Operations" verbose = true begin
            @testset "hadamard" include("core/operations/hadamard.jl")
            @testset "binary_einsum" include("core/operations/binary_einsum.jl")
            @testset "tensor_qr" include("core/operations/tensor_qr.jl")
            @testset "tensor_svd" include("core/operations/tensor_svd.jl")
            @testset "tensor_eigen" include("core/operations/tensor_eigen.jl")
            @testset "simple_update" include("core/operations/simple_update.jl")
        end
        @testset "Tensor" include("core/tensor.jl")
    end
end

if "integration" in target_testsets
    @testset "Integration" verbose = true begin
        @safetestset "OMEinsum" include("integration/omeinsum.jl")
        @safetestset "Strided" include("integration/strided.jl")
        @safetestset "Dagger" include("integration/dagger.jl")
    end
end

if "cuda" in target_testsets
    @safetestset "CUDA" include("integration/cuda.jl")
end

if "reactant" in target_testsets
    @safetestset "Reactant" include("integration/reactant.jl")
end

# using Aqua
# Aqua.test_all(Muscle)
