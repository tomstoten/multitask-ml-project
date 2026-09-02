# Project for Training Multitask Vision Models

I had two ideas while I was riding my bike:

1. Complex models should start simple, and grow more complex over time. Like how human brains do not start fully developed, the model architecture should start underdeveloped and grow while the model is learning. This may help with learning more simple representations of data relevant to the task, and may allow multitask models to find similarities between the tasks.

    - The idea is that before adding complexity to a model, make sure that we already have the best representation of the data/task with a simpler model. That way, it becomes harder to "get lost" in the hyperparameter space, because it starts off small and we repeatedly find the biggest "valley" in the loss-landspace before making it more complex.

2. Complex models should have "sleep" cycles where new pathways are encouraged to form in an unsupervised manner. Similar to how sleep is essential to human brains for solidifying existing connections and forming new ones (even though there is limited/no external stimuli), models should have cycles of activity with no ground-truth labels. 

    - This idea is more complex and less fleshed out right now...

I have no clue if these ideas will work. So I'm going to create a bunch of experiments to find out. Right now, I will focus on image classification tasks since that is what I am familiar with, and the goal will be to see if I can create one model that is good at classifying many types of images from the MedMNIST databank.