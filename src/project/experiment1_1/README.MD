# Experiment 1-1: Single-task, Growing Model - Many vs Few Growth Steps

Now that I have a working implementation of the growing model (which I now know is called Linear Growth Operator or LiGO), I want to see how using different proportion sets impacts the performance.

I hypothesize that by adding more steps, we will be more easily able to find models with fewer parameters that have the same performance as the full-size models. 

At this stage, I need to rethink how I "force" the models to grow.