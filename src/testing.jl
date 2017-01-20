
using Distributions


n = 500
p = 6
# X = hcat(ones(n), randn(n, p-1))



covmat = [1.0 0.6 0.2 0.0 0.0 0.0;
          0.6 1.0 0.6 0.2 0.0 0.0;
          0.2 0.6 1.0 0.6 0.0 0.0;
          0.0 0.2 0.6 1.0 0.0 0.0;
          0.0 0.0 0.0 0.0 1.0 0.0;
          0.0 0.0 0.0 0.0 0.0 1.0]

mvn = MvNormal([1.0, -1.0, -3.0, 3.0, 0.0, 0.0], covmat)

srand(round(Int, time()))
X = rand(mvn, n)'
cor(X)

β = [1.5, 1.1, -10.5, -90.5, 0.0]
η = X*β .+ randn(n)                                          # linear predictor
pr = 1.0 ./ (1.0 + exp(-η))                                  # inv-logit

# simulate outcome variable
y = map(π -> rand(Binomial(1, π)), pr)
mean(y)
find(y)

srand(111)
model, coeffs = build_adaboost_stumps(y, X, 20, subsamp = 1.0);

# apply learned model
y_hat = apply_adaboost_stumps(model, coeffs, X)
mean(y .== y_hat)
1 - mean(y)

# get the probability of each label
# apply_adaboost_stumps_proba(model, coeffs, [5.9, 3.0, 5.1, 1.9, 1.0, 2.1], [0, 1])

srand(111)
model, coeffs = build_adaboost_stumps(y, X, 20, subsamp = 0.8);

y_hat = apply_adaboost_stumps(model, coeffs, randn(p))
mean(y .== y_hat)
1 - mean(y)

# apply_adaboost_stumps_proba(model, coeffs, [5.9, 3.0, 5.1, 1.9, 1.0, 2.1], [0, 1])

find(y)

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
accuracy = nfoldCV_stumps(y, X, 7, 3)
















using DecisionTree
using Distributions

n = 500
p = 4
# X = hcat(ones(n), randn(n, p-1))



covmat = [1.0 0.6 0.2 0.0;
          0.6 1.0 0.6 0.2;
          0.2 0.6 1.0 0.6;
          0.0 0.2 0.6 1.0]

mvn = MvNormal([1.0, -1.0, -3.0, 3.0], covmat)

X = rand(mvn, n)'
cor(X)

β = vcat([1.5, 1.1, -10.5, -100.5])
η = X*β                                                      # linear predictor
pr = 1.0 ./ (1.0 + exp(-η))                                  # inv-logit

# simulate outcome variable
y = map(π -> rand(Binomial(1, π)), pr)
mean(y)

srand(111)
model, coeffs = build_adaboost_stumps(y, X, 20);

# apply learned model
apply_adaboost_stumps(model, coeffs, randn(p))

# get the probability of each label
apply_adaboost_stumps_proba(model, coeffs, [5.9, 3.0, 5.1, 1.9], [0, 1])
