using Test
using Muscle
using Muscle: factordims

@test Invariant' == Invariant
@test Covariant' == Contravariant
@test Contravariant' == Covariant

@test factordims([Covariant, Contravariant]) == ([1], [2])
@test factordims([Contravariant, Covariant]) == ([2], [1])
@test factordims([Covariant, Contravariant, Covariant, Contravariant]) == ([1, 3], [2, 4])
