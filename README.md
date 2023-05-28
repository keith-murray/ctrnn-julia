<div align="center">
<img src="https://raw.githubusercontent.com/ktmurray1999/neural-rules/main/results/figures/FSA_logo.png" alt="logo" width="350"></img>
</div>

# Recurrent networks recognize patterns with low-dimensional oscillations

## What is thie repo?

This repository contains the code for the paper "Recurrent networks recognize patterns with low-dimensional oscillations". All code is in Julia and most is executed via Pluto notebooks. Any data and models generated for the paper have been pregenerated for this repo. All figures for the paper and the LaTeX document are also here.

## What is in the `RecurrentNetworks` directory?

The `RecurrentNetworks` contains all of the code needed for model training. It is a Julia package. There are a few terminal commands required to use the package.

```
julia
using Pkg
Pkg.activate("RecurrentNetworks")
Pkg.status()
Pkg.build()
using RecurrentNetworks
RecurrentNetworks.schedule("./data/setup_data.jld2", "./data/models/", 1)
```