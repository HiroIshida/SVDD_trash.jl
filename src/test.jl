using LinearAlgebra
using Statistics
using JuMP
using OSQP

function wvecmean(vecs; ws = nothing)
    n_dim, n_data = size(vecs)
    isnothing(ws) && (ws = ones(n_data)/n_data)
    ws = repeat(reshape(ws, 1, n_data), n_dim, 1)
    return sum(vecs.*ws, dims = 2)
end

mutable struct SVDD
    X
    kernel
    n_dim
    n_train
    gram
    R
end

function SVDD(X, kernel; C = 0.1)
    n_dim, n_train = size(X)
    gram = _construct_gram_matrix(X, kernel)
    diag_gram = Diagonal(diag(gram))

    # Construct QP problem
    model = Model(with_optimizer(OSQP.Optimizer))
    set_silent(model)
    @variable(model, x[1:n_train])
    @time @objective(model, Max, sum(diag_gram*x) - transpose(x)*gram*x)
    @constraint(model, sum(x) == 1.0)
    @constraint(model, [i=1:n_train], 0<=x[i])
    @constraint(model, [i=1:n_train], x[i]<=C)
    @time JuMP.optimize!(model)
    eps = C * 0.01
    alphas = JuMP.value.(x)
    
    isSupporting(x) = begin 
        x < eps && (return false)
        x > C - eps && (return false)
        return true
    end

    x_center = wvecmean(X; ws = alphas)
    Rs = Float64[]
    for i in 1:n_train
        if isSupporting(alphas[i])
            R_ = norm(X[:, i] - x_center)
            push!(Rs, R_)
        end
    end
    R = mean(Rs)
    return SVDD(X, kernel, n_dim, n_train, gram, R)
end

function _construct_gram_matrix(X, kernel)
    n_dim, n_train = size(X)
    gram = zeros(n_train, n_train)
    for i in 1:n_train
        for j in 1:n_train
            xi, xj = X[:, i], X[:, j]
            gram[i, j] = kernel(xi, xj)
        end
    end
    return gram
end


## main
X = randn(2, 100)
kern = (x, y)-> x'y 
SVDD(X, kern; C = 0.1)

