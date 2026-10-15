# Multitask Learning with Growing Models

### 9/14/26
Just found out that some MIT guys came up with "growing" models 3 years ago. :\(

But it looks like theres only a small number of very recent papers talking about "sleep" cycles in models. I wonder if I can combine them.

[Language Models Need Sleep](https://arxiv.org/pdf/2606.03979) came out 2-3 months ago about sleeping models in LLMs to improve the distillation of information. 

### 9/15/26

The Sleep paper talks about how they try to mimic neuroplasticity by enabling and disabling certain model parameters at different stages of the "sleep" cycle in what they call "Self-Knowledge Seeding", where smaller versions of the model are "distilled" into a larger version of the model. This is basically exactly what my "growing" models are doing, so they got both sleeping and growing down in this paper.

In context, this paper doesn't exactly do multi-task learning in the way I'm proposing, though. LLM's can be argued to be "multi-taskers" because they are good at doing disparate things like math, coding, writing, etc. but these "tasks" are a consequence of the training data, not the model itself (right?). 

- Do LLMs put on different heads to answer different questions? Is that addressed in this paper? (I'm only 5 pages in) 

### 9/28/26

In [Language Models Need Sleep](https://arxiv.org/pdf/2606.03979), the model is pre-trained on a massive dataset (using attention to predict the next token), so the dataset is not a typical input/expected output pair. During the "awake" phase we update the parameters based on user input, whether from the training data or deployment. then during the "sleep" phase, we don't take any input (from training data or from users) and perform self-supervised updates using a student-teacher paradigm.

In the awake phase, training uses two weighted loss terms and splits the model into a teacher (previous snapshot of the model) and student (current model) to encourage distillation of short-term memory from a "wake" phase. The on-policy term lets the student generate a response from the prompt, then grade itself and compare its grade to the teacher's grade to update parameters. The off-policy term lets the student copy the teacher's answer key (training data) to update its latent distribution at each token. This encourages both explicit remembering of past knowledge through the off-policy term and self-improvement through exploration and feedback.

### 10/15/26

Recall that our goal is not to necessarily improve the efficiency of performance, but to find a more optimal set of parameters for multi-task learning, forcing the model to find similarities between the tasks as much as possible while simultaneously finding strong patterns to separate the tasks.

After finishing the methods section of the Sleep paper, I understand mostly how the authors propose to implement "sleep" and "dreaming" into the model. Their core idea is to encourage random parameter updates throughout the sleep cycle (each proposed update comes from a random generation of text called a "dream") by evaluating whether each dream increases performance at a given task or not, and only accept updates if the task performance increases. To me, this is like a random walk through the parameter space with gated acceptances; I would argue that a "dream" should not have the explicit goal of improving performance at a task, and instead should have the goal of encouraging new pathways to form. This could be by saturating connections between neurons that have very low weights, basically forcing the network to use every available connection, although this may just destabalize training and tank performance. Another option is to propose parameter updates that are uncorrelated to "awake" parameter updates. This relies on "two steps forwards, one step back" providing slow but steady convergence, but does nothing to force the model out of local minima in the loss landscape.

As for how this could apply specifically to multitask learning on image models, I need to think more about this...