#####################
# Correctness tests #
#####################

using LdrMatrices
using Test
using LinearAlgebra  


# auxiliary functions for testing
DFT(n) = (1 / sqrt(n)) * [exp(-π * (i - 1) * (j - 1) * 2im / n) for i = 1:n, j = 1:n]
DFT(n, φ) = Diagonal(D(n, φ)) * DFT(n)

@testset "auxiliary tests" begin
    φ = rand(ComplexF64); φ = φ / abs(φ)
    Z_φ = [
        0 0 0 φ
        1 0 0 0
        0 1 0 0
        0 0 1 0
    ]
    V = Diagonal(D(4, φ)) * DFT(4)
    Lambda = Diagonal(Ω(4, φ))
    @test Z_φ ≈ V * Lambda * V'
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
    # TODO
end
