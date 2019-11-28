using LinearAlgebra
using Statistics
using JuMP
using OSQP
using Random

function wvecmean(vecs; ws = nothing)
    n_dim, n_data = size(vecs)
    isnothing(ws) && (ws = ones(n_data)/n_data)
    ws = repeat(reshape(ws, 1, n_data), n_dim, 1)
    return sum(vecs.*ws, dims = 2)
end

mutable struct SVDD
    X
    b_min
    b_max
    kernel
    n_dim
    n_train
    gram
    R
    x_center
    idxes_support
    precomped_term_3rd

    function SVDD(X, kernel; C = 0.1)
        b_min, b_max = get_boundary(X)
        n_dim, n_train = size(X)
        gram = _construct_gram_matrix(X, kernel)
        alphas = _solve_qp(gram, C)
        x_center = wvecmean(X; ws = alphas)
        idxes_support = _get_supporting_indices(alphas, C)
        Rs = [norm(X[:, idx] - x_center) for idx in idxes_support]
        R = mean(Rs)

        # precomp 
        A = repeat(alphas, 1, n_train)
        precomped_term_3rd = A.*A'.*gram
        new(X, b_min, b_max, kernel, n_dim, n_train, gram, R, x_center, idxes_support, precomped_term_3rd)
    end
end

function get_boundary(svdd::SVDD)
    return svdd.b_min, svdd.b_max
end

function predict(svdd::SVDD, x)
    term_1st = x'x
    term_2nd = -2 * sum([svdd.kernel(x, X[:, i]) for i in 1:svdd.n_train])
    term_3rd = sum(svdd.precomped_term_3rd)
    r = sqrt(term_1st + term_2nd + term_3rd)
    return r - svdd.R
end

function get_boundary(X; margin = 0.2) 
  b_min_ = minimum(X; dims = 2)
  b_max_ = maximum(X; dims = 2)
  diff = b_max_ - b_min_
  b_min = vec(b_min_ - diff * 0.2)
  b_max = vec(b_max_ + diff * 0.2)
  return b_min, b_max
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

function _solve_qp(gram, C)
    n_train = size(gram)[1]
    diag_gram = Diagonal(diag(gram))
    model = Model(with_optimizer(OSQP.Optimizer))
    set_silent(model)
    @variable(model, a[1:n_train])
    @objective(model, Max, sum(diag_gram*a) - transpose(a)*gram*a)
    @constraint(model, sum(a) == 1.0)
    @constraint(model, [i=1:n_train], 0<=a[i])
    @constraint(model, [i=1:n_train], a[i]<=C)
    @time JuMP.optimize!(model)
    alphas = JuMP.value.(a)
    return alphas
end

function _get_supporting_indices(alphas, C)
    eps = C * 0.01
    isSupporting(a) = begin 
        a < eps && (return false)
        a > C - eps && (return false)
        return true
    end
    idx_lst = Int64[]
    for (i, alpha) in enumerate(alphas)
        isSupporting(alpha) && push!(idx_lst, i)
    end
    return idx_lst
end

## main
#Random.seed!(0)
X = randn(2, 10) * 0.5
kern = (x, y)-> x'y 
svdd = SVDD(X, kern; C = 10.0)

using Plots
f = (x, y)->predict(svdd, [x, y])
b_min, b_max = get_boundary(svdd::SVDD)
n_contour = 20

x = range(-2.0, stop = 2.0, length = 10)
y = range(-2.0, stop = 2.0, length = 10)
plt = contour(x, y, f, fill=true)
scatter!(X[1, :], X[2, :])
plt

