# Experiment 2.2: Multi-task learning with shared backbone - Comparing Model Growth Parameters

*Change from Experiment 2.2.3*: Growth phases: `[0.1, 1]`

### Results

This shows worse performance for the growing model, likely due to local minimums with a much smaller sub-model becoming harder to escape from. We see that after growing, the losses still stay about the same, much higher than the full model from a random initialization. I think this is because the 10% of pre-trained neurons are encouraging the remaining other 90% to fall in line rather than learning new patterns.