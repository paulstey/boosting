

include("smote_exs.jl")
include("ub_smote.jl")
include("Boosting.jl")
include("simtools.jl")


# running single simulation
@time runsim(500, 20, 2.0, 0.1, seed = 111, ξ = 0.5)

# running multiple simulations
simresults = run_simulations(50, 500, 10, -2.5, 0.05, ξ = 0.6)
summarise_sims(simresults, 5)








n = 100
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

ϵ = rand(Normal(2, 0.5), n)                                  # gives 10% minority class,
# ϵ = rand(Normal(-2.5, 0.5), n)                               # gives 5% minority class,
# ϵ = rand(Normal(4, 0.5), n)                                  # gives 15% minority class,

β = [-3, -4, 1, -5, -4, 0.0, 0.0, 0.0, 0.0, 0.0]
η = X*β + ϵ                                                    # linear predictor w/ error
pr = 1.0 ./ (1.0 + exp(-η))                                    # inv-logit

# simulate outcome variable
y = map(π -> rand(Binomial(1, π)), pr)
y[y .== 0] = -1

# y2, X2 = get_subset(y, X, 450, 0.10)
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
