module MuscleDaggerExt

using Muscle
using Dagger

function __init__()
    Muscle.register_backend!(Muscle.BackendDagger())
    Muscle.register_backend_for_op!(Muscle.binary_einsum, Muscle.BackendDagger())
end

Muscle.platform(::Dagger.DArray) = Muscle.PlatformDagger()

Dagger.domainchunks(t::Tensor) = Dagger.domainchunks(parent(t))

include("binary_einsum.jl")

end
