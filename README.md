<div align="center">
<img src="https://github.com/ktmurray1999/neural-rules/blob/main/results/figures/pca_summary.png" alt="logo" width="400"></img>
</div>

# Recurrent networks recognize patterns with low-dimensional oscillations

## What is thie repo?

This repository contains the code for the paper "Recurrent networks recognize patterns with low-dimensional oscillations". All code is in Julia and most is executed via Pluto notebooks. Any data and models generated for the paper have been pregenerated for this repo. All figures for the paper and the LaTeX document are also here.

## What is in the `RecurrentNetworks` directory?

The `RecurrentNetworks` directory contains all of the code needed for model training. It is a Julia package. There are a few terminal commands required to use the package:

```
julia
using Pkg
Pkg.activate("RecurrentNetworks")
Pkg.status()
Pkg.build()
using RecurrentNetworks
```

To train the model, run the following code:

```
RecurrentNetworks.schedule("./data/setup_data.jld2", "./data/models/", 1)
```

## What is in the `data` directory?

The `data` directory contains all the data required to train the model. Also, the `models` subdirectory contains trained models. Model `model_31.jls` is the moded used to generate figures for the paper. 

`data_540.jls` is the training data file and `data_27.jls` is the testing data file. These were generate via the `training_testing_data.jl` script with the `SETs.csv` file. The `SETs.csv` file was generate via the `generate_data.py` script.

## What is in the `results` directory?

The `results` directory contains figures for the paper and the LaTeX code for the paper. Figures can be found in the `figures` subdirectory and LaTeX code can be found in the `IEEE Submission` subdirectory. The `murray_recurrent_networks.pdf` file is the latest manuscript of the paper.

## What is in the `scripts` directory?

The `scripts` directory contains various scripts to generate data, examine trained models, and generate data and figures. All scripts can be run with Pluto notebooks.

+ `create_scheduler.jl` - Creates a file that contains all the parameters used for training.
+ `examine_saved_models.jl` - Performs some cursory analysis of a trained model.
+ `generate_ieee_figures.jl` - Creates the figures for the paper and stores them in `results/figures`.
+ `training_testing_data.jl` - Creates the data used for training and testing and stores it in `data`.

## Anything else?

I wouldn't recommend using Julia for training recurrent neural networks. In my experience, Julia's automatic differentiation packages don't quite produce accurate gradients; however, they were close enough for this project.
