# Experiment 2.2: Multi-task learning with shared backbone - Comparing Model Growth Parameters

*Change from Experiment 2.2*: Increase epoch budget to 2000 and only grow after 20 epochs of no improvement AFTER this growth phase's validation loss gets below the previous phases's minimum validation loss.

Consider if we should also change the LR scheduler logic for the growing models?

### Results

This had little effect on the outcome. Baseline model still performs significantly better across all datasets, even when allowing the growing model to train for 2000 epochs. I still suspect that this is the result of some poor training hyperparameterization, for example with the learning rate or early stopping. Given enough time, the models should reach the same level of performance because their architectures are identical, so it doesn't make sense to me that the growing models are consistently worse.
