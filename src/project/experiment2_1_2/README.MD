# Experiment 2.1.2: Multi-task learning with shared backbone

*Change from Experiment 2.1*: Changed epochs to interleave batches of single data types instead of mixing them completely. Implemented early stopping and epoch balancing to make sure all data types in a task get equal representation during training.

### Results

This looks like it helped a little bit, but not nearly as much as data augmentation did. I will try using both next, and then move on to another experiment of actual interest.