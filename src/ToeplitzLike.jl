#####################
# Toeplitz matrices #
#####################

struct Toeplitz{Scalar<:Number} <: AbstractMatrix{Scalar}
    coeffs::Vector{Scalar}
    m::Int
    n::Int
    function Toeplitz{Scalar}(coeffs, m, n) where {Scalar<:Number}
        if length(coeffs) == m + n - 1
            new{Scalar}(convert(Vector{Scalar}, coeffs), m, n)
        else
            DimensionMismatch()
        end
    end
end
function Toeplitz{Scalar}(coeff_fc::Vector, coeff_fr::Vector) where {Scalar<:Number}
    m = length(coeff_fc)
    n = length(coeff_fr) + 1
    coeffs = [reverse(coeff_fr); coeff_fc]
    return Toeplitz{Scalar}(coeffs, m, n)
end
Base.:size(A::Toeplitz) = (A.m, A.n)
Base.:getindex(A::Toeplitz, i::Int, j::Int) = A.coeffs[i-j+A.n]
function Base.:+(A::Toeplitz, B::Toeplitz)
    if size(A) != size(B)
        DimensionMismatch()
    else
        coeffs_new = A.coeffs + B.coeffs
        return Toeplitz{eltype(coeffs_new)}(coeffs_new, size(A, 1), size(A, 2))
    end
end


##########################
# Toeplitz-like matrices #
##########################
struct ToeplitzLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    φ::Scalar
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int
    n::Int
    r::Int

    function ToeplitzLike{Scalar}(φ, U, V) where {Scalar<:Number}
        if size(U, 2) != size(V, 2)
            throw(DomainError("size(U,2) != size(V,2)"))
        end
        m = size(U, 1) 
        n = size(V, 1) 
        r = size(U, 2)

        new{Scalar}(
            convert(Scalar, φ),
            convert(Matrix{Scalar}, U),
            convert(Matrix{Scalar}, V),
            m,
            n,
            r,
        )
    end
end
function ToeplitzLike{Scalar}(A::Toeplitz; φ::Number = -1) where {Scalar<:Number}
    U = [
        1 A.coeffs[A.m]
        zeros(A.m - 1) A.coeffs[1:A.m-1]-φ*A.coeffs[A.n+1:end]
    ]
    V = [
        conj(A.coeffs[end:-1:A.m+1] - A.coeffs[A.n-1:-1:1]) zeros(A.n - 1)
        conj(-φ * A.coeffs[A.n]) 1
    ]
    return ToeplitzLike{Scalar}(φ, U, V)
end
Base.:size(A::ToeplitzLike) = (A.m, A.n)
function Base.:getindex(A::ToeplitzLike, i::Int, j::Int)
# TODO
# φ = rand(ComplexF64); φ = φ / abs(φ)
# Z = [
#     0 0 0 0 1
#     1 0 0 0 0
#     0 1 0 0 0
#     0 0 1 0 0
#     0 0 0 1 0
# ]
# Z_φ =  [
#     0 0 0 0 φ
#     1 0 0 0 0
#     0 1 0 0 0
#     0 0 1 0 0
#     0 0 0 1 0
# ]
# u = rand(ComplexF64, 5)
# v = rand(ComplexF64, 5)

# C1 = [u[1] u[5] u[4] u[3] u[2]
#       u[2] u[1] u[5] u[4] u[3]
#       u[3] u[2] u[1] u[5] u[4]
#       u[4] u[3] u[2] u[1] u[5]
#       u[5] u[4] u[3] u[2] u[1]]

# C2 = (1-φ) \ [conj(v[5]) φ*conj(v[1]) φ*conj(v[2]) φ*conj(v[3]) φ*conj(v[4])
#              conj(v[4]) conj(v[5]) φ*conj(v[1]) φ*conj(v[2]) φ*conj(v[3])
#              conj(v[3]) conj(v[4]) conj(v[5]) φ*conj(v[1]) φ*conj(v[2])
#              conj(v[2]) conj(v[3]) conj(v[4]) conj(v[5]) φ*conj(v[1])
#              conj(v[1]) conj(v[2]) conj(v[3]) conj(v[4]) conj(v[5])]
# Tlike = C1*C2
    
# Tliketest = reshape(( kron(Matrix(I,5,5),Z)-  kron(transpose(Z_φ), Matrix(I,5,5))) \ (u * v')[:] ,(5,5))
# Z * Tlike - Tlike * Z_φ ≈ u * v'
    return 0
end


function cauchyform_toeplitz_complex(A::Toeplitz; φ = -1.0 + 0.0im)
    Alike = ToeplitzLike{eltype(A)}(A; φ = φ)
    U = sqrt(Alike.m) .* ifft(Alike.U, 1)
    V = sqrt(Alike.n) .* ifft(Diagonal(D(Alike.n, φ))' * Alike.V, 1)
    return CauchyLike{ComplexF64}(Ω(Alike.m), Ω(Alike.n, φ), U, V)
end

function fast_ge_solve(A::Toeplitz, b::Vector)

    if size(A, 1) == size(A, 2)
        # Solve Cauchy system
        Ahat = cauchyform_toeplitz_complex(A)
        bhat = sqrt(A.n) * ifft(b)
        xhat = fast_ge_solve(Ahat, bhat)

        # retrieve solution of original system
        x = Diagonal(D(A.n, -1.0 + 0.0im)) * fft(xhat) / sqrt(A.n)
    elseif size(A, 1) > size(A, 2)
        error("overdetermined not yet supported")
    else
        error("underdetermined not yet supported")
    end

    return x
end


