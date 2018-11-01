# BayesianOptimization

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/jbrea/BayesianOptimization.jl.svg?branch=master)](https://travis-ci.org/jbrea/BayesianOptimization.jl)
[![codecov.io](http://codecov.io/github/jbrea/BayesianOptimization.jl/coverage.svg?branch=master)](http://codecov.io/github/jbrea/BayesianOptimization.jl?branch=master)

## Usage

```julia
using BayesianOptimization, GaussianProcesses

f(x) = (x - 1).^2 + randn()                      # noisy function to minimize

model = ElasticGPE(2,                            # elastic GP with input dimensions 2
                                                 # The GP is called elastic, because data can be appended efficiently
                   mean = MeanConst(0.),         
                   kernel = SEArd([0., 0.], 5.),
                   logNoise = 0., 
                   capacity = 3000)              # the initial capacity of the GP is 3000 samples.
# Optimize the hyperparameters of the GP using maximum likelihood (ML) estimates every 50 steps
modeloptimizer = MLGPOptimizer(every = 50, noisebounds = [-4, 3],                                     kernbounds = [[-1, -1, 0], [4, 4, 10]],
                               maxeval = 40),
opt = BOpt(f, 
           model,
           ExpectedImprovement(),                 # type of acquisition
           modeloptimizer,                        
           [-5., 0.], [10., 15.],                 # lowerbounds, upperbounds         
           maxiterations = 500, 
           sense = Min, 
           gradientfree = false,                  # use gradient information 
           verbosity = Progress)

result = boptimize!(opt)
```

This package exports 
* `BOpt`, `boptimize!`
* acquisition types: `ExpectedImprovement`, `ProbabilityOfImprovement`, `UpperConfidenceBound`, `ThompsonSamplingSimple`
* scaling of standard deviation in `UpperConfidenceBound`: `BrochuBetaScaling`, `NoBetaScaling`
* GP hyperparameter optimizer: `MLGPOptimizer`, `NoOptimizer`
* optimization sense: `Min`, `Max`
* verbosity levels: `Silent`, `Timings`, `Progress`
Use the REPL help, e.g. `?Bopt`, to get more information.

## Review papers on Bayesian optimization

* [A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning](https://arxiv.org/abs/1012.2599v1)
* [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://ieeexplore.ieee.org/document/7352306)

## Similar Projects

[BayesOpt](https://github.com/jbrea/BayesOpt.jl) is a wrapper of the established
[BayesOpt](https://github.com/rmcantin/bayesopt) toolbox written in C++. It has
more features and seems comparable in performance to this package.

