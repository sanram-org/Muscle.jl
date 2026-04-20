using Test
using Muscle
using Muscle: IndexList, MutableIndexList, findperm, factorinds

il = IndexList([Index(:a), Index(:b), Index(:c)])
@test collect(il) == [Index(:a), Index(:b), Index(:c)]
@test length(il) == 3
@test size(il) == (3,)
@test il[1] == Index(:a)
@test il[2] == Index(:b)
@test il[3] == Index(:c)
@test similar(il) == MutableIndexList{Index}()

mil = MutableIndexList([Index(:a), Index(:b), Index(:c)])
@test collect(mil) == [Index(:a), Index(:b), Index(:c)]
@test length(mil) == 3
@test size(mil) == (3,)
@test mil[1] == Index(:a)
@test mil[2] == Index(:b)
@test mil[3] == Index(:c)
@test similar(mil) == MutableIndexList{Index}()

@test findperm([Index(:a), Index(:b), Index(:c)], [Index(:b), Index(:c), Index(:a)]) == [3, 1, 2]

@test factorinds([Index(:a), Index(:b), Index(:c)], [Index(:a)], [Index(:b), Index(:c)]) == ([Index(:a)], [Index(:b), Index(:c)])
@test factorinds([Index(:a), Index(:b), Index(:c)], [Index(:a)], Index[]) == ([Index(:a)], [Index(:b), Index(:c)])
@test factorinds([Index(:a), Index(:b), Index(:c)], Index[], [Index(:b), Index(:c)]) == ([Index(:a)], [Index(:b), Index(:c)])
@test_throws ArgumentError factorinds([Index(:a), Index(:b), Index(:c)], Index[], Index[])
@test factorinds([Index(:i), Index(:j)], Index[], Index[]) == ([Index(:i)], [Index(:j)])
