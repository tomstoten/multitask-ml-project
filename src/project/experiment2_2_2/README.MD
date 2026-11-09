# Experiment 2.2: Multi-task learning with shared backbone - Comparing Model Growth Parameters

*Change from Experiment 2.2.1*: Make LR scheduler only step after model is fully grown.

If this doesn't work, I will need to consider making changes to the augmentation or architecture itself.

### Results

This change resulted mostly in insignificant differences between the baseline and growing model, although the baseline model still performs marginally better across most categories.

At this point, I am willing to accept that growing the model doesn't necessarily improve convergence. I will run a few more experiments with fewer growth phases since that seemed to work better before, but I'm going to move on to another experiment soon.
