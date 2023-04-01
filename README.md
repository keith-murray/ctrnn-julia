# neural-rules

Accompanying code for "Neural geometry for rule-governed pattern recognition."

## What languages?

Python is used for data generation. String patterns are stored as numbers in a csv file.
Julia is used for model training. Model training scripts are in the form of a Pluto notebook.

## What do the notebooks mean?

The notebooks were used to prototype code before writing it into the `RecurrentNetworks` package. They were constructed in the following order:

1. `visualize_set_data.jl`
    1. Iinitial visualization of the SET data created in python.
2. `modify_neuralode.jl`
    1. First attempt at creating a NeuralODE that takes functions as input.
3. `create_parallel_loader.jl`
    1. Initial prototype of what a dataloader for a batch of NeuralODEs would look like.
    2. This script is more of a prototype of using `ParallelEnsembles`.
4. `probe_neural_ode.jl`
    1. This script is a copy of the NeuralODE tutorial from the Lux wiki.
    2. It was created in order to understand the dimensions of the results.
5. `pilot_neuralode_training.jl`
    1. An initial script that combined all the insights from the previous scripts.
    2. The script included loading SET data, creating function-list types, defining NeuralODEs, and training NeuralODEs.
6. `neuralode_gpu.jl`
    1. This script was an attempt to transition training to CUDA.
    2. It's a work in progress.
7. `neuralode_like_kay.jl`
    1. Certain features of the `pilot_neuralode_training.jl` script were changed to be more like the "Neural dynamics and geometry for transitive inference" paper by Kay et al.
8. `visualize_neuralode_results.jl`
    1. This script is a further iteration of `neuralode_like_kay.jl` with visualizations of the PCA of recurrent trajectories.
9. `training_testing_data.jl`
    1. This script generates the testing and training data that are used for training and testing the NeuralODEs. This script only needs to be run once.
10. `create_scheduler.jl`
    1. This script generates all the training conditions for training NeuralODEs in a principled and reproducable manner.
11. `examine_saved_models.jl`
    1. This script can be used to examine the training and testing accuracies for a trained model. It will also use PCA to visualize the low-dimensional dynamcis.
