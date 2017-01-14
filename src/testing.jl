
using Distributions


n = 200
p = 10
X = hcat(ones(n), randn(n, p-1))

noise_coefs = zeros(p - 5)
β = vcat([1.5, 1.1, -10.5, 4.5, 1.9], noise_coefs)
η = X*β                                                   # linear predictor
pr = 1./(1 + exp(-η))                                     # inv-logit

# simulate outcome variable
y = map(π -> rand(Binomial(1, π)), pr)                    
mean(y)


model, coeffs = build_adaboost_stumps(y, X, 7);

# apply learned model
apply_adaboost_stumps(model, coeffs, randn(p))

# get the probability of each label
apply_adaboost_stumps_proba(model, coeffs, [5.9,3.0,5.1,1.9], ["setosa", "versicolor", "virginica"])

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
accuracy = nfoldCV_stumps(y, X, 7, 3)

