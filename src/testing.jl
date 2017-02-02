
using Distributions

include("Boosting.jl")

# This is a helper function that assures we have the
# proportion of positive cases that we specify.
function curate_data(y, X, n, pct)
    N = length(y)
    if n > N
        error("The input data must have more records than the size of the desired output")
    elseif sum(y)/n < pct
        println(sum(y)/n)
        error("The input data must have more positive cases than desired output percentage")
    end
    pos_indcs = find(y)
    neg_indcs = setdiff(1:N, pos_indcs)
    num_pos = round(Int, n * pct)
    num_neg = n - num_pos
    indcs = vcat(sample(pos_indcs, num_pos), sample(neg_indcs, num_neg))
    return (y[indcs], X[indcs, :])
end


function gen_varcov(p, ρ1, ρ2, ρ3)
    Σ = eye(p)
    Σ[2, 1], Σ[1, 2] = ρ1, ρ1
    Σ[3, 1], Σ[1, 3] = ρ2, ρ2
    Σ[3, 2], Σ[2, 3] = ρ3, ρ3
    Σ[4, 3], Σ[3, 4] = ρ3, ρ3
    Σ[5, 3], Σ[3, 5] = ρ2, ρ2
    Σ[5, 4], Σ[4, 5] = ρ1, ρ1
    return Σ
end


function fitmodels(y, X, train, ntrees, subsample, ξ)
    n = length(y)
    test = setdiff(1:n, train)
    perf = zeros(3, 5)

    for i = 1:3
        if i == 1
            model, coeffs = build_adaboost_stumps(y[train], X[train, :], ntrees, subsample)
        elseif i == 2
            model, coeffs = build_adaboost_stumps_prefer(y[train], X[train, :], ntrees, subsample)
        else i == 3
            model, coeffs = build_adaboost_stumps_weight(y[train], X[train, :], ntrees, subsample, ξ)
        end
        train_err = adaboost_train_error(y[train], X[train, :], model, coeffs)
        y_hat::Array{Int,1} = apply_adaboost_stumps(model, coeffs, X[test, :])
        y_hat_train::Array{Int,1} = apply_adaboost_stumps(model, coeffs, X[train, :])
        perf[i, :] = vcat(1 - train_err, mean(y[test] .== y_hat), roc_auc_score(y[train], y_hat_train), roc_auc_score(y[test], y_hat), f1score(roc(y[test], y_hat)))
    end
    perf
end

# @code_warntype fitmodels([1, 0, 1], randn(3, 3), [true, false, true], 10, 0.7, 0.1)
function simdata(n, p, μ_err)
    Σ = gen_varcov(p, 0.6, 0.3, 0.1)
    mvn = MvNormal(ones(p), Σ)
    N = n + round(Int, n * 0.5)                      # add 50% to desired n, so we can use curate_data()
    X1 = rand(mvn, N)'

    ϵ = rand(Normal(μ_err, 0.5), N)
    β = vcat([-3, -4, 1, -5, -4], zeros(p - 5))
    η = X1*β + ϵ                                     # linear predictor w/ error
    pr = 1.0 ./ (1.0 + exp(-η))                      # inv-logit
    y1 = map(π -> rand(Binomial(1, π)), pr)          # simulate outcome variable
    return (y1, X1)
end


function runsim(n, p, μ_err, pct; ntrees = 100, subsample = 0.7, seed = round(Int, time()), ξ = 0.1)
    srand(seed)
    y = zeros(n)
    X = zeros(n, p)
    # We use this loop to ensure that data curation works. In rare
    # cases we get an error resulting from too few positive cases.
    retry = true
    while retry
        y1, X1 = simdata(n, p, μ_err)
        retry = false

        try
            y, X = curate_data(y1, X1, n, pct)
        catch
            retry = true
        end
    end

    ntrain = round(Int, n * 0.7)
    train = sample(1:n, ntrain, replace = false)

    perf = fitmodels(y, X, train, ntrees, subsample, ξ)
    println("Proportion of minority class in training set: $(mean(y[train]))")
    return perf
end

runsim(100, 10, 2, 0.1, seed = 111, ξ = 0.1)



function run_simulations(nsims, n, p, μ_err, pct, ntrees = 100, subsample = 0.7; ξ = 0.1)
    perf = zeros(3 * nsims, 7)

    simnum = 1
    for i = 1:3:(3*nsims)
        seed = round(Int, time())

        perf[i:i+2, 1] = simnum
        perf[i:i+2, 2] = [1.0, 2.0, 3.0]        # this is the AdaBoost variation
        perf[i:i+2, 3:end] = runsim(n, p, μ_err, pct, ntrees = ntrees, subsample = subsample, seed = seed, ξ = ξ)
        simnum += 1
    end
    return perf
end

simresults = run_simulations(100, 200, 10, 2, 0.1, ξ = 0.1)

function colmeans(X)
    p = size(X, 2)
    res = zeros(p)
    for j = 1:p
        res[j] = mean(X[:, j])
    end
    res
end

function summarise_sims(X)
    means = zeros(3, 6)
    boost_typ = 1.0
    for i = 1:3
        keep = find(x -> x == boost_typ, X[:, 2])
        means[i, 1] = boost_typ
        means[i, 2:end] = colmeans(X[keep, 3:end])
        boost_typ += 1.0
    end
    means
end

summarise_sims(simresults)










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

ϵ = rand(Normal(3, 0.5), n)                                  # gives 10% minority class,
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








using MLBase


























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
