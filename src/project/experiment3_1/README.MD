# Experiment 3.1: Multi-task learning with Jacobian Descent

*Differences from Experiment 3:* Added `torchjd` for Jacobian-informed gradient trajectories to optimize training over many tasks.

In a single iteration through the epoch, a batch is selected from each tasks and run through the backbone + respective classifier in sequence. Then, the losses are aggregated and a single Jacobian is calculated across the losses, rather than a gradient of the sum of losses. Then the Jacbobian is converted to a gradient for normal backpropagation with "adjusted" weights. This prevents conflicting gradient directions from different tasks from cancelling each other out.

*Growth phases*: `[0.1, 0.5, 1]`

### Results

