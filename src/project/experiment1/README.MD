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

Each row represents one `evaluation_metrics_*.json` result file. Size `28` denotes runs whose filename did not include an explicit image size.

## Overall Performance

| Dataset | Size | Model | Timestamp | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Blood | 28 | BaselineModel | 20260909-150658 | 0.9959 | 0.9398 | 0.9293 | 0.9411 | 0.9293 | 0.9342 |
| Blood | 28 | GrowingModel | 20260909-160014 | 0.9958 | 0.9430 | 0.9378 | 0.9382 | 0.9378 | 0.9379 |
| Breast | 28 | BaselineModel | 20260909-151220 | 0.8515 | 0.8397 | 0.7701 | 0.8053 | 0.7701 | 0.7845 |
| Breast | 28 | GrowingModel | 20260909-160618 | 0.8549 | 0.8654 | 0.8327 | 0.8278 | 0.8327 | 0.8302 |
| Derma | 28 | BaselineModel | 20260909-151323 | 0.8086 | 0.7352 | 0.3891 | 0.3880 | 0.3891 | 0.3861 |
| Derma | 28 | GrowingModel | 20260909-160726 | 0.8350 | 0.7571 | 0.3920 | 0.4209 | 0.3920 | 0.4019 |
| Path | 28 | BaselineModel | 20260909-151709 | 0.9677 | 0.8350 | 0.7949 | 0.7989 | 0.7949 | 0.7911 |
| Path | 28 | GrowingModel | 20260909-161116 | 0.9530 | 0.7798 | 0.7190 | 0.6590 | 0.7190 | 0.6787 |

## Classwise Performance

### Blood | size 28 | BaselineModel | 20260909-150658

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9967 | 0.9886 | 0.9522 | 0.9289 | 0.9098 | 0.9193 | 0.9946 | 244.0000 |
| 1 | 0.9996 | 0.9965 | 0.9910 | 0.9984 | 0.9824 | 0.9903 | 0.9996 | 624.0000 |
| 2 | 0.9988 | 0.9912 | 0.9706 | 0.9577 | 0.9453 | 0.9515 | 0.9958 | 311.0000 |
| 3 | 0.9853 | 0.9503 | 0.9309 | 0.8220 | 0.9016 | 0.8600 | 0.9602 | 579.0000 |
| 4 | 0.9976 | 0.9909 | 0.9685 | 0.9309 | 0.9424 | 0.9366 | 0.9947 | 243.0000 |
| 5 | 0.9924 | 0.9784 | 0.8937 | 0.9375 | 0.7923 | 0.8588 | 0.9952 | 284.0000 |
| 6 | 0.9965 | 0.9845 | 0.9756 | 0.9595 | 0.9610 | 0.9602 | 0.9902 | 666.0000 |
| 7 | 1.0000 | 0.9991 | 0.9995 | 0.9937 | 1.0000 | 0.9968 | 0.9990 | 470.0000 |

### Blood | size 28 | GrowingModel | 20260909-160014

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9971 | 0.9898 | 0.9604 | 0.9300 | 0.9262 | 0.9281 | 0.9946 | 244.0000 |
| 1 | 0.9993 | 0.9936 | 0.9898 | 0.9808 | 0.9840 | 0.9824 | 0.9957 | 624.0000 |
| 2 | 0.9969 | 0.9904 | 0.9643 | 0.9603 | 0.9325 | 0.9462 | 0.9961 | 311.0000 |
| 3 | 0.9871 | 0.9579 | 0.9252 | 0.8756 | 0.8756 | 0.8756 | 0.9747 | 579.0000 |
| 4 | 0.9971 | 0.9898 | 0.9622 | 0.9262 | 0.9300 | 0.9281 | 0.9943 | 243.0000 |
| 5 | 0.9923 | 0.9807 | 0.9414 | 0.8759 | 0.8944 | 0.8850 | 0.9885 | 284.0000 |
| 6 | 0.9964 | 0.9845 | 0.9750 | 0.9609 | 0.9595 | 0.9602 | 0.9906 | 666.0000 |
| 7 | 0.9999 | 0.9994 | 0.9997 | 0.9958 | 1.0000 | 0.9979 | 0.9993 | 470.0000 |

### Breast | size 28 | BaselineModel | 20260909-151220

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8517 | 0.8397 | 0.7701 | 0.7429 | 0.6190 | 0.6753 | 0.9211 | 42.0000 |
| 1 | 0.8513 | 0.8397 | 0.7701 | 0.8678 | 0.9211 | 0.8936 | 0.6190 | 114.0000 |

### Breast | size 28 | GrowingModel | 20260909-160618

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.8546 | 0.8654 | 0.8327 | 0.7442 | 0.7619 | 0.7529 | 0.9035 | 42.0000 |
| 1 | 0.8553 | 0.8654 | 0.8327 | 0.9115 | 0.9035 | 0.9075 | 0.7619 | 114.0000 |

### Derma | size 28 | BaselineModel | 20260909-151323

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9169 | 0.9661 | 0.6971 | 0.4821 | 0.4091 | 0.4426 | 0.9850 | 66.0000 |
| 1 | 0.9415 | 0.9406 | 0.7713 | 0.4412 | 0.5825 | 0.5021 | 0.9600 | 103.0000 |
| 2 | 0.8634 | 0.8918 | 0.6961 | 0.5078 | 0.4455 | 0.4746 | 0.9468 | 220.0000 |
| 3 | 0.6490 | 0.9885 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 23.0000 |
| 4 | 0.8140 | 0.8768 | 0.6639 | 0.4394 | 0.3901 | 0.4133 | 0.9377 | 223.0000 |
| 5 | 0.8636 | 0.8209 | 0.7825 | 0.8453 | 0.8963 | 0.8701 | 0.6687 | 1341.0000 |
| 6 | 0.6117 | 0.9855 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 29.0000 |

### Derma | size 28 | GrowingModel | 20260909-160726

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9356 | 0.9631 | 0.7321 | 0.4444 | 0.4848 | 0.4638 | 0.9794 | 66.0000 |
| 1 | 0.9447 | 0.9571 | 0.7616 | 0.5895 | 0.5437 | 0.5657 | 0.9795 | 103.0000 |
| 2 | 0.8715 | 0.9092 | 0.7119 | 0.6159 | 0.4591 | 0.5260 | 0.9647 | 220.0000 |
| 3 | 0.7636 | 0.9885 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 23.0000 |
| 4 | 0.8379 | 0.8843 | 0.6367 | 0.4702 | 0.3184 | 0.3797 | 0.9551 | 223.0000 |
| 5 | 0.8817 | 0.8264 | 0.7695 | 0.8260 | 0.9381 | 0.8785 | 0.6009 | 1341.0000 |
| 6 | 0.6098 | 0.9855 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 29.0000 |

### Path | size 28 | BaselineModel | 20260909-151709

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9871 | 0.9572 | 0.9308 | 0.8827 | 0.8886 | 0.8857 | 0.9730 | 1338.0000 |
| 1 | 0.9887 | 0.9717 | 0.9840 | 0.8067 | 1.0000 | 0.8930 | 0.9679 | 847.0000 |
| 2 | 0.9043 | 0.9440 | 0.7267 | 0.4198 | 0.4867 | 0.4508 | 0.9667 | 339.0000 |
| 3 | 0.9962 | 0.9873 | 0.9624 | 0.9249 | 0.9322 | 0.9285 | 0.9927 | 634.0000 |
| 4 | 0.9609 | 0.9557 | 0.8520 | 0.9812 | 0.7063 | 0.8213 | 0.9977 | 1035.0000 |
| 5 | 0.9340 | 0.9344 | 0.8474 | 0.5797 | 0.7432 | 0.6514 | 0.9516 | 592.0000 |
| 6 | 0.9946 | 0.9844 | 0.9471 | 0.9461 | 0.9001 | 0.9225 | 0.9941 | 741.0000 |
| 7 | 0.9520 | 0.9621 | 0.7883 | 0.7135 | 0.5914 | 0.6468 | 0.9852 | 421.0000 |
| 8 | 0.9914 | 0.9730 | 0.9461 | 0.9355 | 0.9051 | 0.9200 | 0.9871 | 1233.0000 |

### Path | size 28 | GrowingModel | 20260909-161116

| Class | AUC | Accuracy | Balanced Accuracy | Precision | Recall | F1 | Specificity | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9946 | 0.9772 | 0.9742 | 0.9134 | 0.9694 | 0.9405 | 0.9789 | 1338.0000 |
| 1 | 0.9931 | 0.9809 | 0.9892 | 0.8608 | 1.0000 | 0.9252 | 0.9784 | 847.0000 |
| 2 | 0.9484 | 0.9556 | 0.8309 | 0.5222 | 0.6932 | 0.5957 | 0.9686 | 339.0000 |
| 3 | 0.9951 | 0.9713 | 0.9814 | 0.7572 | 0.9937 | 0.8595 | 0.9691 | 634.0000 |
| 4 | 0.9867 | 0.9630 | 0.9093 | 0.9018 | 0.8338 | 0.8665 | 0.9847 | 1035.0000 |
| 5 | 0.9015 | 0.9487 | 0.8306 | 0.6892 | 0.6892 | 0.6892 | 0.9721 | 592.0000 |
| 6 | 0.8600 | 0.8968 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 741.0000 |
| 7 | 0.9240 | 0.9478 | 0.6615 | 0.5966 | 0.3373 | 0.4310 | 0.9858 | 421.0000 |
| 8 | 0.9738 | 0.9184 | 0.9327 | 0.6895 | 0.9546 | 0.8007 | 0.9109 | 1233.0000 |
