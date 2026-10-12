# Experiment 2: Multi-task learning with shared backbone

This experiments seeks to show whether growing models have a discernable edge over baseline models in multi-task learning. The same model backbone (convolutional encoder + fully connected distiller) produces a consistently-sized feature vector for all tasks, and only the classifier heads will differ. The goal is to have one model that can look at a Blood sample, a Derma image, and a Path image, and classify each of them into their respective sublasses.

### Multi-task design

Each dataset will serve as a single classification task; because datasets have differing number of classes to identify, we use different classifier heads, although we construct these heads to be as simple as possible for the sake of testing the internal representation of data. The backbone must learn to use similar features across differing image types to perform well across datasets.

We will have to partition the datasets by their size and number of channels, as these parameters fundamentally change the network architecture. Therefore, Blood/Derma/Path (color, 3 channels) will be grouped together, and Breast/Tissue (black and white, 1 channel) will be grouped together.

I will leave the door open for future addition of more datasets, but in this initial experiment it will suffice to have a simple 2-task 1-channel data partition and a more complex 3-task 3-channel partition.

### Implementation

Refactoring the dataloaders was inevitable, so I guess now is the time to do it. I think it best to move dataloaders inside of experiments to let me tinker with them for each new experiment without ruining the implementation for previous experiments. 

I also want to try getting faster dataloading by using memory-mapping and lazy loading.

### Results

It looks like many of these multi-task models are overfitting: validation curves are going up consistently, while training curves move low or stay low.

I think I will try again with a decrease LR on plateau scheduler and a little bit of weight decay on the optimizer. Then future experiments can compare growth schedules here.