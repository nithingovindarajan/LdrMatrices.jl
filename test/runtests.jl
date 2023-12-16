#####################
# Correctness tests #
#####################

using LdrMatrices
using Test
using LinearAlgebra  
using FFTW
    
# auxiliary functions for testing
DFT(n) = (1 / sqrt(n)) .* [exp(-π * (i - 1) * (j - 1) * 2im / n) for i = 1:n, j = 1:n]
DFT(n, φ) = Diagonal(D(n, φ)) * DFT(n)
DST_I(n) = sqrt(2/(n+1))  * [sin((k*j*π)/(n+1)) for k = 1:n, j = 1:n]
function DCT_II(n)
    A = sqrt(2/n)  * [cos(((2*k-1)*(j-1)*π)/(2*n)) for k = 1:n, j=1:n]
    sqrt_2 = sqrt(2)
    A[:,1] = A[:,1] ./ sqrt_2 
    return A
end 


@testset "auxiliary tests" begin
    # relation trigonemetric transforms and FFTW code
    n = 10
    @test DFT(n) ≈ 1/sqrt(n)*fft(Matrix(I,n,n),1)
    #@test DST_I(n) ≈ bfft(Matrix(I,n,n),1) ???
    @test DCT_II(n) ≈ idct(Matrix(I,n,n),1)
    
    # eigendecomposition fo Z_phi
    φ = rand(ComplexF64); φ = φ / abs(φ)
    Z_φ = [
        0 0 0 φ
        1 0 0 0
        0 1 0 0
        0 0 1 0
    ]
    V = Diagonal(D(4, φ)) * DFT(4)
    Λ = Diagonal(Ω(4, φ))
    @test Z_φ ≈ V * Λ * V'
        
    # eigendecomposition of Y_00
    Y_00 = [0 1 0 0 0
            1 0 1 0 0
            0 1 0 1 0
            0 0 1 0 1
            0 0 0 1 0]
    V = DST_I(5)
    Λ = Diagonal(Ξ(5))
    @test Y_00 ≈ V * Λ * V'
   
    # eigendecomposition of Y_11       
    Y_11 = [1 1 0 0 0
            1 0 1 0 0
            0 1 0 1 0
            0 0 1 0 1
            0 0 0 1 1]
    V = DCT_II(5)
    Λ = Diagonal(Κ(5))
    @test Y_11 ≈ V * Λ * V'    
end

@testset "construct Cauchy-like matrices" begin
    n = 8
    omega, lambda = Ω(n, 1), Ω(n, -1 + 0im)
    U, V = rand(n, 2), rand(n, 2)
    Acauchy = CauchyLike{ComplexF64}(omega, lambda)
    Acauchylike = CauchyLike{ComplexF64}(omega, lambda, U, V)
    @test Matrix(Acauchy) ≈ 1 ./ (omega .- transpose(lambda))
    @test Matrix(Acauchylike) ≈ (U * V') .* Acauchy
    @test Matrix(Acauchylike) ≈ getindex(Acauchylike, 1:n, 1:n)
end

@testset "GKO algorithm LU decomposition" begin
    n = 10
    omega, lambda = Ω(n, 1), Ω(n, -1 + 0im)
    U, V = rand(n, 2), rand(n, 2)
    Acauchylike = CauchyLike{ComplexF64}(omega, lambda, U, V)
    Π_1, Π_2, L, U = fast_LU_cauchy(Acauchylike, row_pivot = true, column_pivot = true)
    @test Acauchylike[Π_1, Π_2] ≈ L * U
end

@testset "GKO algorithm solve" begin
    n = 10
    omega, lambda = Ω(n, 1), Ω(n, -1 + 0im)
    U, V = rand(n, 2), rand(n, 2)
    Acauchylike = CauchyLike{ComplexF64}(omega, lambda, U, V)
    b = rand(ComplexF64, n)
    x = Acauchylike \ b
    x_ref = Matrix(Acauchylike) \ b
    @test x ≈ x_ref
end

@testset "Toeplitz matrices" begin
    u = rand(4)
    v = rand(3)
    b = rand(ComplexF64, 4)
    Tdense = [
        u[1] v[1] v[2] v[3]
        u[2] u[1] v[1] v[2]
        u[3] u[2] u[1] v[1]
        u[4] u[3] u[2] u[1]
    ]
    T = Toeplitz{ComplexF64}(u, v)
    @test T == Tdense
    Z = [
        0 0 0 1
        1 0 0 0
        0 1 0 0
        0 0 1 0
    ]
    Z_min1 = copy(Z)
    Z_min1[1, 4] = -1
    U, V = LdrMatrices.ldr_generators_I(T)
    @test Z*T - T*Z_min1 ≈ U * V'
    @test LdrMatrices.cauchyform_complex(T) ≈ DFT(4)' * Tdense * DFT(4, -1.0 + 0.0im)
    @test (Tdense \ b) ≈ T \ b
end

@testset "Hankel matrices" begin
    b = rand(ComplexF64, 4)
    h1 = rand(ComplexF64, 4)
    h2 = rand(ComplexF64, 3)
    h = [h1;h2]
    Hdense = [h[1] h[2] h[3] h[4]
              h[2] h[3] h[4] h[5]
              h[3] h[4] h[5] h[6]
              h[4] h[5] h[6] h[7]]
    H = Hankel{ComplexF64}(h,4,4)
    @test H == Hdense
    @test H == Hankel{ComplexF64}(h1,h2)                        
end

@testset "Toeplitz-plus-Hankel matrices" begin
    
    # Toeplitz part
    u = rand(4)
    v = rand(3)
    
    Tdense = [
        u[1] v[1] v[2] v[3]
        u[2] u[1] v[1] v[2]
        u[3] u[2] u[1] v[1]
        u[4] u[3] u[2] u[1]
    ]
    T = Toeplitz{ComplexF64}(u, v)

    # hankel part
    h1 = rand(ComplexF64, 4)
    h2 = rand(ComplexF64, 3)
    h = [h1;h2]
    Hdense = [h[1] h[2] h[3] h[4]
              h[2] h[3] h[4] h[5]
              h[3] h[4] h[5] h[6]
              h[4] h[5] h[6] h[7]]
    H = Hankel{ComplexF64}(h,4,4)

    #
    TplusH = ToeplitzPlusHankel{ComplexF64}(u,v,h1,h2)
    @test TplusH  == Tdense + Hdense
    @test typeof(T+H) == typeof(H+T) == typeof(TplusH)
    @test H+T == T+H == TplusH 

end
