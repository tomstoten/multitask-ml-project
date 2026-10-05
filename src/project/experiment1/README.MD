# Experiment 1: Single-task, Full model vs. Growing model

This first experiment will force me to actually figure out how to implement a growing model. I don't think PyTorch will play nicely with dynamically adding parameters to an existing model. But I guess we'll find out...

## Growing Model Implementation

I've opted to "Grow" my models by defining a maximum model size and fixed architecture, but by masking out certain neuron activations AND gradients, I have effectively created a variable-size model. I define a list of proportions that say how much of each layer should actually work; e.g. proportions = (0.1, 0.5, 1) means that when we initialize our model, each layer (convolutional or fully connected) will have only its first 10% of neurons "awake", and the rest are initialized to 0 with gradients masked out. Once I want to grow the model, I will initialize the next 40% of the neurons (to reach a total awake proportion of 0.5) using standard random init while keeping the weights of the already trained neurons the same. In the last growth step, the remaining 50% of neurons are initialized and gradient unmasked. 

I've decided for now that growth steps should occur when the model has reached a relative equilibrium with its current parameters, i.e. our validation performance plateaus. Also, when we are 80% of the way through the maximum training epochs, we will force the model to grow to full size in one step. This will most likely change in future experiments, but I had to make a decision now for the sake of getting on with the thing.

The final classifier layer was moved into a separate module becuase I think in the future, we will want to specify just the classifier in terms of the data set, allowing the model to have a consistent feature vector size for all data sets (i.e. all tasks). My theory is that this model growth will be especially good at training multi-task models. This menas that in order to consistently grade performance, we will be using pre-trained classifiers (which are a single layer + softmax = multinomial logistic regression) and only training the model weights to output a feature vector.

## Experiment 1

I tested the same model architecture on 5 different MedMNIST data sets (Blood, Breast, Derma, Path, Tissue), with the only independent variable being wether the model grows or not. The growth proportions were emperically set to (0.1, 0.5, 1), and the image sizes of 28 and 224 were tested for all data sets.

### Results

# Evaluation Metrics Summary

Each row represents one `evaluation_metrics_*.json` result file.

## Overall Performance

| Dataset | Size | Model | Timestamp | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blood | 224 | BaselineModel | September 14, 2026 | 0.9981 | 0.9678 | 0.9692 | 0.9681 | 0.9692 | 0.9684 |
| Blood | 224 | GrowingModel | September 16, 2026 | 0.9966 | 0.9500 | 0.9532 | 0.9411 | 0.9532 | 0.9467 |
| Blood | 28 | BaselineModel | September 9, 2026 | 0.9959 | 0.9398 | 0.9293 | 0.9411 | 0.9293 | 0.9342 |
| Blood | 28 | GrowingModel | September 9, 2026 | 0.9958 | 0.9430 | 0.9378 | 0.9382 | 0.9378 | 0.9379 |
| Breast | 224 | BaselineModel | September 14, 2026 | 0.8808 | 0.8462 | 0.7970 | 0.8065 | 0.7970 | 0.8015 |
| Breast | 224 | GrowingModel | September 16, 2026 | 0.8689 | 0.8654 | 0.8026 | 0.8421 | 0.8026 | 0.8190 |
| Breast | 28 | BaselineModel | September 9, 2026 | 0.8515 | 0.8397 | 0.7701 | 0.8053 | 0.7701 | 0.7845 |
| Breast | 28 | GrowingModel | September 9, 2026 | 0.8549 | 0.8654 | 0.8327 | 0.8278 | 0.8327 | 0.8302 |
| Derma | 224 | BaselineModel | September 14, 2026 | 0.8759 | 0.7157 | 0.4077 | 0.4736 | 0.4077 | 0.4224 |
| Derma | 224 | GrowingModel | September 16, 2026 | 0.8679 | 0.7267 | 0.3939 | 0.4919 | 0.3939 | 0.4196 |
| Derma | 28 | BaselineModel | September 9, 2026 | 0.8086 | 0.7352 | 0.3891 | 0.3880 | 0.3891 | 0.3861 |
| Derma | 28 | GrowingModel | September 9, 2026 | 0.8350 | 0.7571 | 0.3920 | 0.4209 | 0.3920 | 0.4019 |
| Path | 224 | BaselineModel | September 15, 2026 | 0.9720 | 0.8415 | 0.8310 | 0.8198 | 0.8310 | 0.8154 |
| Path | 224 | GrowingModel | September 16, 2026 | 0.9832 | 0.8809 | 0.8620 | 0.8485 | 0.8620 | 0.8509 |
| Path | 28 | BaselineModel | September 9, 2026 | 0.9677 | 0.8350 | 0.7949 | 0.7989 | 0.7949 | 0.7911 |
| Path | 28 | GrowingModel | September 9, 2026 | 0.9530 | 0.7798 | 0.7190 | 0.6590 | 0.7190 | 0.6787 |
| Tissue | 224 | BaselineModel | September 15, 2026 | 0.8161 | 0.5701 | 0.4224 | 0.4212 | 0.4224 | 0.4197 |
| Tissue | 224 | GrowingModel | September 16, 2026 | 0.8099 | 0.5646 | 0.4220 | 0.4167 | 0.4220 | 0.4149 |
| Tissue | 28 | BaselineModel | September 14, 2026 | 0.8037 | 0.6166 | 0.4623 | 0.4119 | 0.4623 | 0.4331 |
| Tissue | 28 | GrowingModel | September 14, 2026 | 0.8109 | 0.6222 | 0.4551 | 0.4160 | 0.4551 | 0.4333 |

## Analysis

This is where I will want to run automated statistical tests to identify if there is a significant difference in performance for AUC (and maybe Bal. Acc.?) metrics for the hypothesis test of interest: 

$H_0 := \theta_0 = \theta_1$ vs $H_1 := \theta_0 \neq \theta_1$

At a glance, it looks like there is no significant difference in performance across all tests, but that may not be the case...