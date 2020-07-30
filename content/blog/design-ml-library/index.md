---
title: Design of a high-level PyTorch ML library
date: 2020-07-30
subject: ["pytorch", "ml", "sw", "design", "opensource"]
author: RSP
description: Description of a high-level ML library based on PyTorch
---

This article is an attempt to think about all the various activities in
a deep-learning based ML model development process and to distill those
into a high-level functional and object-oriented design library based on
the PyTorch framework.

At a high level, the process of ML model development for a single task
can be subdivided into the following stages

1.  Data Setup and Feature Preprocessing: perhaps multiple datasets,
    augmentations,

2.  Model and Objective/Loss Specification: specified in code in PyTorch

3.  Model Training: optimization

4.  Performance Evaluation: performance metrics on validation set, test
    set

The goals of the new high-level library would be to support and simplify

1.  Model specification: modular ability to combine multiple models like
    legos

2.  Loss-function specification and tuning: each layer can have
    different and multiple objectives, can arrive from models themselves

3.  Model deployment tools: quantization, factorization, ONNX etc.

4.  Self-supervised learning and coupled with data augmentation: support
    for online self-supervised learning, test-time self-supervised
    learning

5.  Multi-task learning: train on multiple datasets and tasks. Some
    tasks can share common representations and have task specific
    branches subsequent to that

6.  Curriculum learning: a teacher (RL agent, perhaps) figures out
    sequence of tasks and examples within those tasks to learn better
    representations and/or perform better at tasks

7.  Multi-mode learning: when inputs are from two or more different
    domains (like audio, sensors, video, images, etc.) and ability to
    share/relate representations of same object

8.  ML model debugging: feature attribution, hard examples,
    out-of-domain indicators

9.  Model report card

Let’s take all of these goals and then try to create an example that
allow easy use and expression of high-level ideas quickly.

## High-level example using JumpML library

```python
import jumpml
import torch.optim as optim

# datasetFile defines
#         multiple datasets, train, test, validation splits
#         feature processing pipelines (could include pretrained models themselves)
#         transformations and augmentations
#         curriculum learning options
dataloader = jumpml.createDataloader(datasetFile)

# modelFile defines
#          pytorch modules and connections
#          objective functions at/between layers/nodes
model = jumpml.loadModel(modelFile)

# define optimizer as usual.
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

# Create Learner based on learnerOptions
#          initialization, EMA parameters, EWC, parameter saving
#          self-supervised learning phases
#          curriculum learning based on learning
#          dynamic model augmentation: adding more capacity or branches dynamically
#         logging learning progress
learner = jumpml.createLearner(dataloader, model, optimizer, learnerOptions)

# Self-supervised phase
learner.ssLearn()

# Supervised Learning Phase: perhaps trains only the task-specific layers... may be does some multi-task learning on some other layers.
for epochs in range(numEpochs):
    learner.trainEpoch()  # this could only impact the classifier branches


# Evaluation
# evalMetrics are a list of functions.
# each function takes model predictions, ground truth and calculates metrics
evalResults = jumpml.evalModel(dataloader, model, evalMetrics)

# Reports, visualization, deployment
# may run a separate distillation loop (quantization, factorization, onnx exporting)
jumpml.deployModel(evalResults, model, deploymentConfig)
```
