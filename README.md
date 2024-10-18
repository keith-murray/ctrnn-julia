# CT-RNN Implementation in Julia's SciML Ecosystem

This repository contains an implementation of continuous-time recurrent neural networks (CT-RNNs) in the [Julia programming language](https://julialang.org) and [SciML ecosystem](https://sciml.ai). Specifically, the architecture and training of CT-RNNs is implemented via the [Lux.jl](https://lux.csail.mit.edu/stable/), [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/), and [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) packages.

<div align="center">
<img src="https://github.com/keith-murray/ctrnn-julia/blob/main/results/figures/pca_summary.png" alt="logo" width="400"></img>
</div>

## Installation

Clone the repo and execute the following commands:
```julia
using Pkg
Pkg.activate("RecurrentNetworks")
Pkg.status()
Pkg.build()
using RecurrentNetworks
```

## Usage

To train the model on a sample task, execute the following code:
```julia
RecurrentNetworks.schedule("./data/setup_data.jld2", "./data/models/", 1)
```

## Examples

The `scripts` directory contains various example scripts train and examine trained models. All scripts can be executed with Pluto notebooks.

+ `create_scheduler.jl` - initializes parameter file for training
+ `examine_saved_models.jl` - exploratory analysis of a trained model
+ `training_testing_data.jl` - data initialization for sample task

## A technical note

I wouldn't recommend using Julia for training CT-RNNs. My initial impression was that Julia and the SciML ecosystem could train CT-RNNs faster than PyTorch or TensorFlow in Python; however, I've since found the [JAX ecosystem](https://jax.readthedocs.io/en/latest/) in Python to be strictly better than Julia and SciML. Checkout [`keith-murray/ctrnn-jax`](https://github.com/keith-murray/ctrnn-jax) for a JAX implementation of CT-RNNs that is faster and more usable than this repo.
