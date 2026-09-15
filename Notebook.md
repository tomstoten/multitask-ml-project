# Multitask Learning with Growing Models

### 9/14/26
Just found out that some MIT guys came up with "growing" models 3 years ago. :\(

But it looks like theres only a small number of very recent papers talking about "sleep" cycles in models. I wonder if I can combine them.

[Language Models Need Sleep](https://arxiv.org/pdf/2606.03979) came out 2-3 months ago about sleeping models in LLMs to improve the distillation of information. 

### 9/15/26

The Sleep paper talks about how they try to mimic neuroplasticity by enabling and disabling certain model parameters at different stages of the "sleep" cycle in what they call "Self-Knowledge Seeding", where smaller versions of the model are "distilled" into a larger version of the model. This is basically exactly what my "growing" models are doing, so they got both sleeping and growing down in this paper.

In context, this paper doesn't exactly do multi-task learning in the way I'm proposing, though. LLM's can be argued to be "multi-taskers" because they are good at doing disparate things like math, coding, writing, etc. but these "tasks" are a consequence of the training data, not the model itself (right?). 

- Do LLMs put on different heads to answer different questions? Is that addressed in this paper? (I'm only 5 pages in) 

