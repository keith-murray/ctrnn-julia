<div align="center">
<img src="https://github.com/keith-murray/ctrnn-julia/blob/main/results/figures/pca_summary.png" alt="logo" width="400"></img>
</div>

# CT-RNN Implementation in Julia's SciML Ecosystem

This repository contains an implementation of continuous-time recurrent neural networks (CT-RNNs) in the [Julia programming language](https://julialang.org) and [SciML ecosystem](https://sciml.ai). Specifically, I implemented the architecture and training of CT-RNNs with the [Lux.jl](https://lux.csail.mit.edu/stable/), [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/), and [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/) packages.

## `RecurrentNetworks` directory

The `RecurrentNetworks` directory contains all of the code needed for model training. There are a few terminal commands required to use the package:

```
julia
using Pkg
Pkg.activate("RecurrentNetworks")
Pkg.status()
Pkg.build()
using RecurrentNetworks
```

To train the model, execute the following code:

```
RecurrentNetworks.schedule("./data/setup_data.jld2", "./data/models/", 1)
```

## `data` directory

The `data` directory contains all the data required to train the model. The `models` subdirectory contains trained models.

`data_540.jls` is the training data file and `data_27.jls` is the testing data file. These were generate via the `training_testing_data.jl` script and `SETs.csv` file.

## `scripts` directory

The `scripts` directory contains various scripts to generate data and examine trained models. All scripts can be run with Pluto notebooks.

+ `create_scheduler.jl` - Creates a file containing all the parameters used for training.
+ `examine_saved_models.jl` - Performs exploratory analysis of a trained model.
+ `generate_ieee_figures.jl` - Creates and stores figures in `results/figures`.
+ `training_testing_data.jl` - Creates and stores training data in the `data` directory.

## A technical note

I wouldn't recommend using Julia for training CT-RNNs. My initial impression was that Julia and the SciML ecosystem could train CT-RNNs quicker than PyTorch or TensorFlow in Python; however, I've since found the [JAX ecosystem](https://jax.readthedocs.io/en/latest/) in Python to be significantly faster than Julia. Checkout my [attract-or-oscillate repository](https://github.com/keith-murray/attract-or-oscillate) where I was able to train [16,128 RNNs](https://openreview.net/forum?id=ql3u5ITQ5C) on the [MIT SuperCloud HPC](https://doi.org/10.1109/HPEC.2018.8547629) in about 60 hours.
