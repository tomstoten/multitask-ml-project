# Experiment 1-1: Single-task, Growing Model - Many vs Few Growth Steps

Now that I have a working implementation of the growing model (which I now know is called Linear Growth Operator or LiGO), I want to see how using different proportion sets impacts the performance.

I hypothesize that by adding more steps, we will be more easily able to find models with fewer parameters that have the same performance as the full-size models. 

At this stage, I need to rethink how I "force" the models to grow.

Also, should I add some automatic tests to compare results? 

# Results

Based on the metrics summaries, it looks like there isn't really a significant performance difference in the final full-sized model based on how large the growth steps were. 

The next step could be to see how well we can maximize performance while minimizing parameter usage. 