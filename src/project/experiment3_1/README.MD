# Experiment 3.1: Multi-task learning with Jacobian Descent

*Differences from Experiment 3:* Added `torchjd` for Jacobian-informed gradient trajectories to optimize training over many tasks.

In a single iteration through the epoch, a batch is selected from each tasks and run through the backbone + respective classifier in sequence. Then, the losses are aggregated and a single Jacobian is calculated across the losses, rather than a gradient of the sum of losses. Then the Jacbobian is converted to a gradient for normal backpropagation with "adjusted" weights. This prevents conflicting gradient directions from different tasks from cancelling each other out.

*Growth phases*: `[0.1, 0.5, 1]`

### Results

This looks like it is having small impacts on results, mainly by increasing baseline performance in larger image sizes and decreasing growing model performance, with some variability between datasets. It is also clear that some of the new datasets face severe mode collapse due to unbalanced output classes, in particular, Chest, Vessel, and Synapse. I will need to address this issue by adjusting the sampling algorithm to equitably train on all output classes.

I want to try a couple things: first, I want to perform actual logistic regression on the backbone outputs for testing instead of using the pre-trained classifier weights to make sure we are actually getting the best fit possible from the features we have. Then, if this shows promise, I would consider doing the same for validation since what we really care about is the model's shared representation, not the actual metrics from each training run. Second, I want to change my growth operator to be something that actually does something other than creating new randomly initialized nodes.