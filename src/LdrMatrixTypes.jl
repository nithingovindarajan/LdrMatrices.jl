###############
# DEFINITIONS #
###############
D(n, φ) = [φ^(-(k - 1) / n) for k = 1:n]
Ω(n) = [exp(π * 2im * (k - 1) / n) for k = 1:n]
Ω(n, φ) = φ^(1 / n) * Ω(n)
Ξ(n) = [2 * cos((k * π) / (n + 1)) for k = 1:n]
Κ(n) = [2 * cos((k * π) / (n)) for k = 0:n-1]

########################
# Cauchy-Like matrices #
########################
struct CauchyLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    omega::Vector{Scalar}
    lambda::Vector{Scalar}
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int
    n::Int
    r::Int

    function CauchyLike{Scalar}(omega, lambda, U, V) where {Scalar<:Number}
        if size(U, 2) != size(V, 2)
            throw(DomainError("size(U,2) != size(V,2)"))
        end
        if size(U, 1) != length(omega)
            throw(DomainError("size(U,1) != length(omega)"))
        end
        if size(V, 1) != length(lambda)
            throw(DomainError("size(V,1) != length(lambda)"))
        end
        m = length(omega)
        n = length(lambda)
        r = size(U, 2)

        new{Scalar}(
            convert(Vector{Scalar}, omega),
            convert(Vector{Scalar}, lambda),
            convert(Matrix{Scalar}, U),
            convert(Matrix{Scalar}, V),
            m,
            n,
            r,
        )
    end

end
function CauchyLike{Scalar}(omega, lambda) where {Scalar<:Number}
    return CauchyLike{Scalar}(
        omega,
        lambda,
        ones(length(omega), 1),
        ones(length(lambda), 1),
    )
end
Base.:size(A::CauchyLike) = (A.m, A.n)
Base.:getindex(A::CauchyLike, i::Int, j::Int) =
    @views dot(A.V[j, :], A.U[i, :]) / (A.omega[i] - A.lambda[j])
Base.:Matrix(A::CauchyLike{Scalar}) where {Scalar<:Number} =
    ((A.U * A.V') ./ (A.omega .- transpose(A.lambda)))



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






###################
# Hankel matrices #
###################
struct Hankel{Scalar<:Number} <: AbstractMatrix{Scalar}
    coeffs::Vector{Scalar}
    m::Int
    n::Int
    function Hankel{Scalar}(coeffs, m, n) where {Scalar<:Number}
        if length(coeffs) == m + n - 1
            new{Scalar}(convert(Vector{Scalar}, coeffs), m, n)
        else
            DimensionMismatch()
        end
    end
end
function Hankel{Scalar}(coeff_fr::Vector, coeff_lc::Vector) where {Scalar<:Number}
    m = length(coeff_fr)
    n = length(coeff_lc) + 1
    coeffs = [coeff_fr; coeff_lc]
    return Hankel{Scalar}(coeffs, m, n)
end
Base.:size(A::Hankel) = (A.m, A.n)
Base.:getindex(A::Hankel, i::Int, j::Int) = A.coeffs[i+j-1]
function Base.:+(A::Hankel, B::Hankel)
    if size(A) != size(B)
        DimensionMismatch()
    else
        coeffs_new = A.coeffs + B.coeffs
        return Hankel{eltype(coeffs_new)}(coeffs_new, size(A, 1), size(A, 2))
    end
end


########################
# Hankel-like matrices #
########################
struct HankelLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    φ::Scalar
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int
    n::Int
    r::Int

    function HankelLike{Scalar}(φ, U, V) where {Scalar<:Number}
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
function HankelLike{Scalar}(A::Hankel; φ::Number = -1) where {Scalar<:Number}
    U = [
        A.coeffs[A.n+1:end]-φ*A.coeffs[1:A.m-1] zeros(A.m - 1)
        -φ*A.coeffs[A.m] 1
    ]
    V = [
        zeros(A.n - 1) conj(A.coeffs[1:A.n-1] - A.coeffs[A.m+1:end])
        1 conj(A.coeffs[A.n])
    ]
    return HankelLike{Scalar}(φ, U, V)
end
Base.:size(A::ToeplitzLike) = (A.m, A.n)
function Base.:getindex(A::ToeplitzLike, i::Int, j::Int)
# TODO
    return 0
end



#################################
# Toeplitz-plus-Hankel matrices #
#################################
struct ToeplitzPlusHankel{Scalar<:Number} <: AbstractMatrix{Scalar}
    coeffs_toeplitz::Vector{Scalar}
    coeffs_hankel::Vector{Scalar}
    m::Int
    n::Int
    function ToeplitzPlusHankel{Scalar}(
        coeffs_toeplitz,
        coeffs_hankel,
        m,
        n,
    ) where {Scalar<:Number}
        if length(coeffs_toeplitz) == length(coeffs_hankel) == m + n - 1
            new{Scalar}(
                convert(Vector{Scalar}, coeffs_toeplitz),
                convert(Vector{Scalar}, coeffs_hankel),
                m,
                n,
            )
        else
            DimensionMismatch()
        end
    end

end
function ToeplitzPlusHankel{Scalar}(
    coeff_toeplitz_fc::Vector,
    coeff_toeplitz_fr::Vector,
    coeff_hankel_fr::Vector,
    coeff_hankel_lc::Vector,
) where {Scalar<:Number}
    m = length(coeff_toeplitz_fc)
    n = length(coeff_toeplitz_fr) + 1
    coeffs_toeplitz = [reverse(coeff_toeplitz_fr); coeff_toeplitz_fc]
    coeffs_hankel = [coeff_hankel_fr; coeff_hankel_lc]
    return ToeplitzPlusHankel{Scalar}(coeffs_toeplitz, coeffs_hankel, m, n)
end
Base.:size(A::ToeplitzPlusHankel) = (A.m, A.n)
Base.:getindex(A::ToeplitzPlusHankel, i::Int, j::Int) =
    A.coeffs_hankel[i+j-1] + A.coeffs_toeplitz[i-j+A.m]
function Base.:+(A::Toeplitz, B::Hankel)
    if size(A) != size(B)
        DimensionMismatch()
    else
        elem_type = typeof(A[1, 1] + B[1, 1])
        return ToeplitzPlusHankel{elem_type}(A.coeffs, B.coeffs, size(A, 1), size(A, 2))
    end
end
function Base.:+(A::Hankel, B::Toeplitz)
    if size(A) != size(B)
        DimensionMismatch()
    else
        elem_type = typeof(A[1, 1] + B[1, 1])
        return ToeplitzPlusHankel{elem_type}(B.coeffs, A.coeffs, size(A, 1), size(A, 2))
    end
end
# TODO more toeplitzplushankel additions


#################################
# Toeplitz-Hankel-like matrices #
#################################
struct ToeplitzPlusHankelLike{Scalar<:Number} <: AbstractMatrix{Scalar}
    U::Matrix{Scalar}
    V::Matrix{Scalar}
    m::Int
    n::Int
    r::Int

    function ToeplitzPlusHankelLike{Scalar}(U, V) where {Scalar<:Number}
        if size(U, 2) != size(V, 2)
            throw(DomainError("size(U,2) != size(V,2)"))
        end
        m = size(U, 1) 
        n = size(V, 1) 
        r = size(U, 2)

        new{Scalar}(
            convert(Matrix{Scalar}, U),
            convert(Matrix{Scalar}, V),
            m,
            n,
            r,
        )
    end
end
function ToeplitzPlusHankelLike{Scalar}(A::Toeplitz; φ::Number = -1) where {Scalar<:Number}
# associated with lrd eq: Y00 * A -  A * Y11 = UV'
    # U
    u1 = [
        1
        zeros(A.m - 1)
    ]
    u2 = [
        zeros(A.m - 1)
        1
    ]
    u3 = [
        -A.coeffs[1]
        A.coeffs[1:A.m-1] - A.coeffs[2:A.m]
    ]
    u4 = [
        A.coeffs[end-A.m+2:end] - A.coeffs[end-A.m+1:end-1]
        -A.coeffs[end]
    ]
    U = [u1 u2 u3 u4]
    # V    
    v1 = [
        -conj(A.coeffs[A.n-1:-1:1])
        0
    ]
    v2 = [
        0
        -conj(A.coeffs[end:-1:A.m+1])
    ]
    v3 = [
        zeros(A.n - 1)
        1
    ]
    v4 = [
        1
        zeros(A.n - 1)
    ]
    V = [v1 v2 v3 v4]
    return ToeplitzPlusHankelLike{Scalar}(U, V)
end
function ToeplitzPlusHankelLike{Scalar}(A::Hankel; φ::Number = -1) where {Scalar<:Number}
    # U
    u1 = [
        1
        zeros(A.m - 1)
    ]
    u2 = [
        zeros(A.m - 1)
        1
    ]
    u3 = [
        A.coeffs[end-A.m+2:end] - A.coeffs[end-A.m+1:end-1]
        -A.coeffs[end]
    ]
    u4 = [
        -A.coeffs[1]
        A.coeffs[1:A.m-1] - A.coeffs[2:A.m]
    ]
    U = [u1 u2 u3 u4]
    # V      
    v1 = [
        0
        -conj(A.coeffs[1:A.n-1])
    ]
    v2 = [
        -conj(A.coeffs[end-A.n+2:end])
        0
    ]
    v3 = [
        zeros(A.n - 1)
        1
    ]
    v4 = [
        1
        zeros(A.n - 1)
    ]
    V = [v1 v2 v3 v4]
    return ToeplitzPlusHankelLike{Scalar}(U, V)
end
function ToeplitzPlusHankelLike{Scalar}(A::ToeplitzPlusHankel; φ::Number = -1) where {Scalar<:Number}
    # U
    u1 = [
        1
        zeros(A.m - 1)
    ]
    u2 = [
        zeros(A.m - 1)
        1
    ]
    u3 = [
        A.coeffs_hankel[end-A.m+2:end] - A.coeffs_hankel[end-A.m+1:end-1]
        -A.coeffs_hankel[end]
    ]
    u3 += [
        -A.coeffs_toeplitz[1]
        A.coeffs_toeplitz[1:A.m-1] - A.coeffs_toeplitz[2:A.m]
    ]
    u4 = [
        -A.coeffs_hankel[1]
        A.coeffs_hankel[1:A.m-1] - A.coeffs_hankel[2:A.m]
    ]
    u4 += [
        A.coeffs_toeplitz[end-A.m+2:end] - A.coeffs_toeplitz[end-A.m+1:end-1]
        -A.coeffs_toeplitz[end]
    ]
    U = [u1 u2 u3 u4]
    # V      
    v1 = [
        0
        -conj(A.coeffs_hankel[1:A.n-1])
    ]
    v1 += [
        -conj(A.coeffs_toeplitz[A.n-1:-1:1])
        0
    ]
    v2 = [
        -conj(A.coeffs_hankel[end-A.n+2:end])
        0
    ]
    v2 += [
        0
        -conj(A.coeffs_toeplitz[end:-1:A.m+1])
    ]
    v3 = [
        zeros(A.n - 1)
        1
    ]
    v4 = [
        1
        zeros(A.n - 1)
    ]
    V = [v1 v2 v3 v4]
    return ToeplitzPlusHankelLike{Scalar}(U, V)
end
Base.:size(A::ToeplitzPlusHankelLike) = (A.m, A.n)
function Base.:getindex(A::ToeplitzPlusHankelLike, i::Int, j::Int)
# TODO
    return 0
end
