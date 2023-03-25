# BayesianOptimization

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/jbrea/BayesianOptimization.jl.svg?branch=master)](https://travis-ci.com/jbrea/BayesianOptimization.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/github/jbrea/BayesianOptimization.jl?branch=master&svg=true)](https://ci.appveyor.com/project/jbrea/bayesianoptimization-jl)
[![codecov.io](http://codecov.io/github/jbrea/BayesianOptimization.jl/coverage.svg?branch=master)](http://codecov.io/github/jbrea/BayesianOptimization.jl?branch=master)

## Usage

```julia
using BayesianOptimization, GaussianProcesses, Distributions

f(x) = sum((x .- 1).^2) + randn()                # noisy function to minimize

# Choose as a model an elastic GP with input dimensions 2.
# The GP is called elastic, because data can be appended efficiently.
model = ElasticGPE(2,                            # 2 input dimensions
                   mean = MeanConst(0.),         
                   kernel = SEArd([0., 0.], 5.),
                   logNoise = 0.,
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
set_priors!(model.mean, [Normal(1, 2)])

# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
modeloptimizer = MAPGPOptimizer(every = 50, noisebounds = [-4, 3],       # bounds of the logNoise
                                kernbounds = [[-1, -1, 0], [4, 4, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                maxeval = 40)
opt = BOpt(f,
           model,
           UpperConfidenceBound(),                   # type of acquisition
           modeloptimizer,                        
           [-5., -5.], [5., 5.],                     # lowerbounds, upperbounds         
           repetitions = 5,                          # evaluate the function for each input 5 times
           maxiterations = 100,                      # evaluate at 100 input positions
           sense = Min,                              # minimize the function
           acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                                 restarts = 5,       # run the NLopt method from 5 random initial conditions each time.
                                 maxtime = 0.1,      # run the NLopt method for at most 0.1 second each time
                                 maxeval = 1000),    # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
            verbosity = Progress)

result = boptimize!(opt)
```

### Resume optimization

To continue the optimization, one can call `boptimize!(opt)` multiple times.
```julia
result = boptimize!(opt) # first time (includes initialization)
result = boptimize!(opt) # restart
maxiterations!(opt, 50)  # set maxiterations for the next call
result = boptimize!(opt) # restart again
```

### (Warm-)start with some known function values

By default, the first `5*length(lowerbounds)` input points are sampled from a
Sobol sequence. If instead one has already some function values available and
wants to skip the initialization with the Sobol sequence, one can update the
model with the available data and set `initializer_iterations = 0`. For example
(continuing the above example after setting the `modeloptimizer`).
```julia
x = [rand(2) for _ in 1:20]
y = -f.(x)
append!(model, hcat(x...), y)

opt = BOpt(f,
           model,
           UpperConfidenceBound(),
           modeloptimizer,                        
           [-5., -5.], [5., 5.],
           maxiterations = 100,
           sense = Min,
           initializer_iterations = 0
          )

result = boptimize!(opt)
```

This package exports
* `BOpt`, `boptimize!`, `optimize`
* acquisition types: `ExpectedImprovement`, `ProbabilityOfImprovement`, `UpperConfidenceBound`, `ThompsonSamplingSimple`, `MutualInformation`
* scaling of standard deviation in `UpperConfidenceBound`: `BrochuBetaScaling`, `NoBetaScaling`
* GP hyperparameter optimizer: `MAPGPOptimizer`, `NoModelOptimizer`
* Initializer: `ScaledSobolIterator`, `ScaledLHSIterator`
* optimization sense: `Min`, `Max`
* verbosity levels: `Silent`, `Timings`, `Progress`
* helper: `maxduration!`, `maxiterations!`

Use the REPL help, e.g. `?Bopt`, to get more information.

## Review papers on Bayesian optimization

* [A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning](https://arxiv.org/abs/1012.2599v1)
* [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://ieeexplore.ieee.org/document/7352306)

## Similar Projects

[BayesOpt](https://github.com/jbrea/BayesOpt.jl) is a wrapper of the established
[BayesOpt](https://github.com/rmcantin/bayesopt) toolbox written in C++.

[Dragonfly](https://github.com/dragonfly/dragonfly) is a feature-rich package
for scalable Bayesian optimization written in Python. Use it in Julia with
[PyCall](https://github.com/JuliaPy/PyCall.jl).
