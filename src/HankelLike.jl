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
Base.:size(A::HankelLike) = (A.m, A.n)
function Base.:getindex(A::HankelLike, i::Int, j::Int)
# TODO
    return 0
end


function cauchyform_hankel_complex(A::Hankel; φ = -1.0 + 0.0im)
    Alike = HankelLike{eltype(A)}(A; φ = φ)
    Uhat = sqrt(Alike.m) .* ifft(Alike.U, 1)
    Vhat = sqrt(Alike.n) .* ifft(Diagonal(D(Alike.n, φ))' * Alike.V, 1)
    return CauchyLike{Complex}(conj(Ω(Alike.m)), Ω(Alike.n, φ), Uhat, Vhat)
end

function fast_ge_solve(A::Hankel, b::Vector)
    if size(A, 1) == size(A, 2)
        # Solve Cauchy system
        Ahat = cauchyform_hankel_complex(A)
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