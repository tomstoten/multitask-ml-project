# Experiment 3: Multi-task learning with more data

The goal of this experiment is to compare results over a larger joint task (i.e. more different datasets combined) to see if the results from Experiment 2 hold with more than 3 datasets in one task. This will also include 3D datasets to see if more complex data affects this or not.

*Datasets added:*

| Dataset | Dimensionality | Image sizes | Channels | Classes
| --- | --- | --- | --- | --- |
| Adrenal | 3D | 28 and 64 | 1 | 2 |
| Chest | 2D | 28 and 224 | 1 | 14 binary labels |
| Fracture | 3D | 28 and 64 | 1 | 3 |
| Nodule | 3D | 28 and 64 | 1 | 2 |
| OCT | 2D | 28 and 224 | 1 | 4 |
| Organ3D| 3D | 28 and 64 | 1 | 11 |
| OrganA | 2D | 28 and 224 | 1 | 11 |
| Pneumonia | 2D | 28 and 224 | 1 | 2 |
| Retina | 2D | 28 and 224 | 3 (RGB) | 5 |
| Synapse | 3D | 28 and 64 | 1 | 2 |
| Vessel | 3D | 28 and 64 | 1 | 2 |

*Growth phases*: `[0.1, 0.5, 1]`

### Results

It looks like adding more data gives the growing model a slight (although statistically insignificant) edge over the baseline model for larger images sizes (> 28 pixels/voxels). This leads me to believe that there may be a true relationship between data size / task complexity (i.e. # tasks for one model backbone), although I belive a more complex model architecture will interact with data size to show this more clearly.

The clearest overfitting is in the 2D RGB group at 224 and Chest growing at 224; the 3D 28 group also develops a late gap. The 3D 64 group has little train/validation gap but weak test scores, so it looks more like underfitting or a resolution/generalization problem than classic overfitting. The 2D grayscale group fits most steadily, though Tissue and OCT remain comparatively weak.

## Forward thoughts

I'm considering adding either some sort of latent regularization to help with getting more consistent performance across datasets within a task, or adding a transformer submodule to the architecture as a way to increase complexity and (hopefully) get better model fits.

I could also implement Jacobian Descent to help optimize the backbone better.