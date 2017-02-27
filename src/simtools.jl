
# This is a helper function that assures we have the
# proportion of positive cases that we specify.
function curate_data(y, X, n::Int, pct::Float64)
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


function gen_varcov(p::Int, ρ1::Float64, ρ2::Float64, ρ3::Float64)
    Σ = eye(p)
    Σ[2, 1], Σ[1, 2] = ρ1, ρ1
    Σ[3, 1], Σ[1, 3] = ρ2, ρ2
    Σ[3, 2], Σ[2, 3] = ρ3, ρ3
    Σ[4, 3], Σ[3, 4] = ρ3, ρ3
    Σ[5, 3], Σ[3, 5] = ρ2, ρ2
    Σ[5, 4], Σ[4, 5] = ρ1, ρ1
    return Σ
end

# @code_warntype gen_varcov(10, 0.1, 0.1, 0.3)


function fitmodels{T<:Real}(y::Array{T, 1}, X::Array{T, 2}, train::BitArray{1}, ntrees::Int, subsample::Float64, ξ::Float64)
    n = length(y)
    n_conditions = 5
    test = setdiff(1:n, train)
    perf = zeros(n_conditions, 5)

    for i = 1:n_conditions
        if i == 1
            model, coeffs = build_adaboost_stumps(y[train], X[train, :], ntrees, subsample)
        elseif i == 2
            model, coeffs = build_adaboost_stumps_prefer(y[train], X[train, :], ntrees, subsample)
        elseif i == 3
            model, coeffs = build_adaboost_stumps_weight(y[train], X[train, :], ntrees, subsample, ξ)
        elseif i == 4
            model, coeffs = build_adaboost_stumps_wghtpref(y[train], X[train, :], ntrees, subsample, ξ)
        elseif i == 5
            X_smoted, y_smoted  = smote(X[train, :], y[train], 5)
            model, coeffs = build_adaboost_stumps(y_smoted, X_smoted, ntrees, subsample)
        end
        train_err = adaboost_train_error(y[train], X[train, :], model, coeffs)
        y_hat::Array{Int,1} = apply_adaboost_stumps(model, coeffs, X[test, :])
        y_hat_train::Array{Int,1} = apply_adaboost_stumps(model, coeffs, X[train, :])

        # train_err, test_err, train_auc, test_auc, f1score
        perf[i, :] = vcat(1 - train_err, mean(y[test] .== y_hat), roc_auc_score(y[train], y_hat_train), roc_auc_score(y[test], y_hat), f1score(roc(y[test], y_hat)))
    end
    perf
end

# @code_warntype fitmodels(sample([0.0, 1.0], 100), rand(100, 10), bitrand(100), 10, 0.7, 0.5)


# @code_warntype fitmodels([1, 0, 1], randn(3, 3), [true, false, true], 10, 0.7, 0.1)
function simdata(n::Int, p::Int, μ_err::Float64)
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

# @code_warntype simdata(100, 10, 2)

function runsim(n::Int, p::Int, μ_err::Float64, pct::Float64; ntrees::Int = 100, subsample::Float64 = 0.7, seed::Int = round(Int, time()), ξ::Float64 = 0.1)
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
            warn("Have to retry, curation failed...")
        end
    end

    ntrain = round(Int, n * 0.7)
    train = sample(1:n, ntrain, replace = false)

    perf = fitmodels(y, X, train, ntrees, subsample, ξ)
    println("Proportion of minority class in training set: $(mean(y[train]))")
    return perf
end

# runsim(200, 10, 2, 0.1, seed = 111, ξ = 0.1)



function run_simulations(nsims::Int, n::Int, p::Int, μ_err::Float64, pct::Float64, ntrees = 100, subsample = 0.7; ξ = 0.1)
    n_conditions = 5
    n_trials = n_conditions * nsims
    perf = zeros(n_trials, 7)

    step_size = n_conditions - 1                                            # for stepping through output mat.
    simnum = 1
    for i = 1:n_conditions:n_trials
        seed = round(Int, time())

        perf[i:i+step_size, 1] = simnum
        perf[i:i+step_size, 2] = linspace(1.0, n_conditions, n_conditions)  # indicates the AdaBoost variation
        perf[i:i+step_size, 3:end] = runsim(n,
                                            p,
                                            μ_err,
                                            pct,
                                            ntrees = ntrees,
                                            subsample = subsample,
                                            seed = seed,
                                            ξ = ξ)

        simnum += 1
    end
    return perf
end

# simresults = run_simulations(500, 200, 10, 2, 0.1, ξ = 0.1)

function colmeans(X)
    p = size(X, 2)
    res = zeros(p)
    for j = 1:p
        res[j] = mean(X[:, j])
    end
    res
end


function summarise_sims(X, n_conditions = 5)

    means = zeros(n_conditions, 6)
    boost_typ = 1.0
    for i = 1:n_conditions
        keep = find(x -> x == boost_typ, X[:, 2])
        means[i, 1] = boost_typ
        means[i, 2:end] = colmeans(X[keep, 3:end])
        boost_typ += 1.0
    end
    means
end

# summarise_sims(simresults)
