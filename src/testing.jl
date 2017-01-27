
using Distributions

include("Boosting.jl")


function runsim(n, p, μ_err, ntrees = 50, subsample = 0.7, seed = round(Int, time()))
    srand(seed)
    ρ1 = 0.6
    ρ2 = 0.3
    ρ3 = 0.1
    Σ = eye(p)
    Σ[2, 1], Σ[1, 2] = ρ1, ρ1
    Σ[3, 1], Σ[1, 3] = ρ2, ρ2
    Σ[3, 2], Σ[2, 3] = ρ3, ρ3
    Σ[4, 3], Σ[3, 4] = ρ3, ρ3
    Σ[5, 3], Σ[3, 5] = ρ2, ρ2
    Σ[5, 4], Σ[4, 5] = ρ1, ρ1
    mvn = MvNormal(ones(p), Σ)
    X = rand(mvn, n)'

    ntrain = round(Int, n * 0.7)

    ϵ = rand(Normal(μ_err, 0.5), n)

    β = vcat([-3, -4, 1, -5, -4], zeros(p - 5))
    η = X*β + ϵ                                         # linear predictor w/ error
    pr = 1.0 ./ (1.0 + exp(-η))                         # inv-logit
    y = map(π -> rand(Binomial(1, π)), pr)              # simulate outcome variable

    train = sample(1:n, ntrain, replace = false)
    test = setdiff(1:n, train)

    model, coeffs = build_adaboost_stumps(y[train], X[train, :], ntrees, subsample)
    train_err = adaboost_train_error(y[train], X[train, :], model, coeffs)
    y_hat = apply_adaboost_stumps(model, coeffs, X[test, :])


    println("Proportion of minority class: $(mean(y[train]))")
    return (1 - train_err, mean(y[test] .== y_hat))
end

runsim(500, 10, 1.6)



n = 500
p = 10
# X = hcat(ones(n), randn(n, p-1))



Σ = [1.0 0.8 0.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
     0.8 1.0 0.6 0.2 0.0 0.0 0.0 0.0 0.0 0.0;
     0.4 0.6 1.0 0.6 0.4 0.0 0.0 0.0 0.0 0.0;
     0.0 0.2 0.6 1.0 0.8 0.0 0.0 0.0 0.0 0.0;
     0.0 0.0 0.4 0.8 1.0 0.0 0.0 0.0 0.0 0.0;
     0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
     0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
     0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]

mvn = MvNormal(ones(p), Σ)

srand(round(Int, time()))
X = rand(mvn, n)'
cor(X)

ϵ = rand(Normal(1.6, 0.5), n)                                  # gives 10% minority class,
# ϵ = rand(Normal(-2.5, 0.5), n)                               # gives 5% minority class,
# ϵ = rand(Normal(4, 0.5), n)                                  # gives 15% minority class,

β = [-3, -4, 1, -5, -4, 0.0, 0.0, 0.0, 0.0, 0.0]
η = X*β + ϵ                                                    # linear predictor w/ error
pr = 1.0 ./ (1.0 + exp(-η))                                    # inv-logit

# simulate outcome variable
y = map(π -> rand(Binomial(1, π)), pr)
push!(w1, mean(y))
mean(w1)


find(y)

srand(111)
model, coeffs = build_adaboost_stumps(y, X, 50, 1.0);          # 50 boosting rounds, "sub-sample" 100%

# apply learned model
y_hat = apply_adaboost_stumps(model, coeffs, X)
mean(y .== y_hat)
mean(y)



# get the probability of each label
# apply_adaboost_stumps_proba(model, coeffs, [5.9, 3.0, 5.1, 1.9, 1.0, 2.1], [0, 1])

srand(111)
model, coeffs2 = build_adaboost_stumps(y, X, 50, 1.0, 0.2);

y_hat = apply_adaboost_stumps(model, coeffs2, X)
mean(y .== y_hat)
1 - mean(y)

# apply_adaboost_stumps_proba(model, coeffs, [5.9, 3.0, 5.1, 1.9, 1.0, 2.1], [0, 1])

find(y)

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
# accuracy = nfoldCV_stumps(y, X, 7, 3)





using DataFrames

ds = readtable("example_data.csv");
y = convert(Vector, ds[:, 1]);
X = convert(Array, ds[:, 2:end]);

































using DecisionTree
using Distributions

n = 500
p = 4
# X = hcat(ones(n), randn(n, p-1))



Σ = [1.0 0.6 0.2 0.0;
          0.6 1.0 0.6 0.2;
          0.2 0.6 1.0 0.6;
          0.0 0.2 0.6 1.0]

mvn = MvNormal([1.0, -1.0, -3.0, 3.0], Σ)

X = rand(mvn, n)'
cor(X)

β = vcat([1.5, 1.1, -10.5, -100.5])
η = X*β                                                      # linear predictor
pr = 1.0 ./ (1.0 + exp(-η))                                  # inv-logit

# simulate outcome variable
y = map(π -> rand(Binomial(1, π)), pr)
mean(y)

srand(111)
model, coeffs = DecisionTree.build_adaboost_stumps(y, X, 50);

# apply learned model
y_hat = DecisionTree.apply_adaboost_stumps(model, coeffs, X)
mean(y .== y_hat)
mean(y)

# get the probability of each label
apply_adaboost_stumps_proba(model, coeffs, [5.9, 3.0, 5.1, 1.9], [0, 1])
