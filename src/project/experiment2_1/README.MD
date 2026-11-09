# Experiment 2.1: Multi-task learning with shared backbone

*Change from Experiment 2*: Decrease LR on plateau scheduler and a little bit of weight decay on the optimizer. 

### Results

These results still indicate no significant difference between baseline and growing models for any of the task groups/image sizes, but it is still clear that we are overfitting. I will attempt to fix this in the next experiment by balancing training data between datasets in each epoch and cutting off training after validation peaks rather than allowing the model to run the full 100 epochs.