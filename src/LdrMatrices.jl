module LdrMatrices


###########
# exports #
###########
export Ω, D, Ξ, Κ
export CauchyLike, Toeplitz, Hankel, ToeplitzPlusHankel, ToeplitzLike, HankelLike, ToeplitzPlusHankelLike
export fast_ge_LU, fast_ge_triangularize, fast_ge_inverse, fast_ge_solve


############
# packages #
############
using LinearAlgebra
using FFTW


include("definitions.jl")
include("CauchyLike.jl")
include("ToeplitzLike.jl")
include("HankelLike.jl")
include("ToeplitzPlusHankelLike.jl")

end
