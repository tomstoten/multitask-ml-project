# Experiment 2.2: Multi-task learning with shared backbone - Comparing Model Growth Parameters

*Change from Experiment 2.1.3*: Use growth proportions `[0.25, 0.5, 0.75, 1]` instead of `[0.1, 0.5, 1.0]`. Increase epoch budget to 400.

### Results

This growth scheme lead to significantly worse performance across nearly all metrics, tasks, and datasets compared to the baseline model. It seems that for these large multitask runs, the growing model actually gets stuck in a local minimum and the spike at each growth stage doesn't give the model the "push" it needs to find the new minimum in the larger loss-landscape. Another issue may be that each stage is not given enough epochs to improve performance, so the next experiment will be increasing this budget both per growth phase and overall.

I hypothesize that by adding more parameters relative to the current parameter proportion forces the model to use less of the previously learned information, thus discouraging remaining in the same local minimum. 