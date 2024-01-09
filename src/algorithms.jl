#############################################################################
# Gohberg, Kailath and Olshevsky (Schur) Algorithm for Cauchy-like matrices #
#############################################################################

function normalize(U, V)
    F = qr(U)
    Unorm = F.Q * Matrix(I, size(U)...)
    Vnorm = V * F.R'
    return Unorm, Vnorm
end

#TODO allow for nonsquare
function fast_ge_LU(A::CauchyLike; row_pivot = true, column_pivot = true, TOL = 1E-15)

    # dimensions
    n = A.n
    r = A.r

    # initialize
    Π_1 = collect(Int64, 1:n)
    Π_2 = collect(Int64, 1:n)
    μ = copy(A.omega)
    ν = copy(A.lambda)
    U, V = normalize(A.U, A.V)
    nrms = norm.(eachrow(V))
    a = Vector{eltype(A)}(undef, n)
    LU = Matrix{eltype(A)}(undef, n, n)

    for k = 1:n-1

        #### Pivot columns ####

        # find column pivot
        if column_pivot
            j = argmax(nrms[k:end])
            j += k - 1
        else
            j = k
        end

        # swap entries
        if j != k
            LU[1:(k-1), k], LU[1:(k-1), j] = LU[1:(k-1), j], LU[1:(k-1), k]
            V[k, :], V[j, :] = V[j, :], V[k, :]
            ν[k], ν[j] = ν[j], ν[k]
            Π_2[k], Π_2[j] = Π_2[j], Π_2[k]
        end

        #### Pivot rows ####

        # retrieve first column
        a[k:end] = (U[k:end, :] * conj(V[k, :])) ./ (μ[k:end] .- ν[k])

        # find row pivot
        if row_pivot
            i = argmax(abs.(a[k:end]))
            i += k - 1
        else
            i = k
        end

        # raise warning in case pivot entry is very small
        if abs(a[k]) < TOL
            @warn "Pivot entry is dangerously small!"
        end

        # swap entries
        if i != k
            a[k], a[i] = a[i], a[k]
            U[k, :], U[i, :] = U[i, :], U[k, :]
            μ[k], μ[i] = μ[i], μ[k]
            Π_1[k], Π_1[i] = Π_1[i], Π_1[k]
            LU[k, 1:(k-1)], LU[i, 1:(k-1)] = LU[i, 1:(k-1)], LU[k, 1:(k-1)]
        end

        #### Schur update ####

        # TODO: Givens rotations

        # compute multiplier
        alphinv = 1 / a[k]

        # update LU decomposition 
        LU[k, k] = a[k]
        LU[(k+1):end, k] = alphinv * a[(k+1):end]
        LU[k, (k+1):end] = (conj(V[k+1:end, :]) * U[k, :]) ./ (μ[k] .- ν[k+1:end])

        # Compute generators of schur complement
        U[(k+1):end, :] = U[(k+1):end, :] - LU[(k+1):end, k] * transpose(U[k, :])
        V[(k+1):end, :] =
            V[(k+1):end, :] - conj(alphinv) * conj(LU[k, (k+1):end]) * transpose(V[k, :])

        # normalize generators
        if n - k > r
            U[(k+1):end, :], V[(k+1):end, :] = normalize(U[(k+1):end, :], V[(k+1):end, :])
        end

        # compute norms
        nrms[k+1:end] = norm.(eachrow(V[k+1:end, :]))
    end
    # compute last entry of U
    LU[n, n] = dot(V[n, :], U[n, :]) / (μ[n] - ν[n])

    # triangular factors
    lower = UnitLowerTriangular(LU)
    upper = UpperTriangular(LU)

    return Π_1, Π_2, lower, upper

end


function fast_ge_triangularize(
    A::CauchyLike,
    b::Vector;
    row_pivot = true,
    column_pivot = true,
    TOL = 1E-15,
)
    # check if system is square
    if size(A, 1) != size(A, 2)
        error("overdetermined not yet supported")
    end

    # dimensions
    n = A.n
    r = A.r

    # initialize
    Π_1 = collect(Int64, 1:n)
    Π_2 = collect(Int64, 1:n)

    μ = copy(A.omega)
    ν = copy(A.lambda)
    U, V = normalize(A.U, A.V)
    nrms = norm.(eachrow(V))
    a = Vector{eltype(A)}(undef, n)
    btilde = copy(b)
    Upper = UpperTriangular(Matrix{eltype(A)}(undef, n, n)) # must be a better way!

    for k = 1:n-1

        #### Pivot columns ####

        # find column pivot
        if column_pivot
            j = argmax(nrms[k:end])
            j += k - 1
        else
            j = k
        end

        # swap entries
        if j != k
            Upper[1:(k-1), k], Upper[1:(k-1), j] = Upper[1:(k-1), j], Upper[1:(k-1), k]
            V[k, :], V[j, :] = V[j, :], V[k, :]
            ν[k], ν[j] = ν[j], ν[k]
            Π_2[k], Π_2[j] = Π_2[j], Π_2[k]
        end

        #### Pivot rows ####

        # retrieve first column
        a[k:end] = (U[k:end, :] * conj(V[k, :])) ./ (μ[k:end] .- ν[k])

        # find row pivot
        if row_pivot
            i = argmax(abs.(a[k:end]))
            i += k - 1
        else
            i = k
        end

        # raise warning in case pivot entry is very small
        if abs(a[k]) < TOL
            @warn "Pivot entry is dangerously small!"
        end

        # swap entries
        if i != k
            a[k], a[i] = a[i], a[k]
            U[k, :], U[i, :] = U[i, :], U[k, :]
            μ[k], μ[i] = μ[i], μ[k]
            Π_1[k], Π_1[i] = Π_1[i], Π_1[k]
            btilde[k], btilde[i] = btilde[i], btilde[k]
        end

        #### Schur update ####

        # TODO: first do givens rotations

        # compute multiplier
        alphinv = 1 / a[k]
        f = alphinv * a[(k+1):end]

        # update Upper and btilde
        Upper[k, k] = a[k]
        btilde[(k+1):end] = btilde[(k+1):end] - f * btilde[k]
        Upper[k, (k+1):end] = (conj(V[k+1:end, :]) * U[k, :]) ./ (μ[k] .- ν[k+1:end])

        # Compute generators of schur complement
        U[(k+1):end, :] = U[(k+1):end, :] - f * transpose(U[k, :])
        V[(k+1):end, :] =
            V[(k+1):end, :] - conj(alphinv) * conj(Upper[k, (k+1):end]) * transpose(V[k, :])

        # normalize generators
        if n - k > r
            U[(k+1):end, :], V[(k+1):end, :] = normalize(U[(k+1):end, :], V[(k+1):end, :])
        end

        # compute norms
        nrms[k+1:end] = norm.(eachrow(V[k+1:end, :]))
    end
    # compute last entry of U
    Upper[n, n] = dot(V[n, :], U[n, :]) / (μ[n] - ν[n])



    return Upper, Π_2, btilde
end

function fast_ge_inverse(A::CauchyLike, row_pivot = true, column_pivot = true)
    #TODO 
end

function fast_ge_solve(A::CauchyLike, b::Vector)

    if size(A, 1) == size(A, 2)
        # triangularize
        Upper, Π_2, btilde = fast_ge_triangularize(A, b)
        # solve triangular system
        x = Upper \ btilde
        # inverse permutation
        x[Π_2] = x
    elseif size(A, 1) > size(A, 2)
        error("overdetermined not yet supported")
    else
        error("underdetermined not yet supported")
    end

    return x
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
