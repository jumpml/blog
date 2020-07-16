---
title: Perspectives on machine-learned intelligence
date: 2020-07-16
subject: ["ml", "perspectives"]
author: RSP
description: The characteristics of intelligent systems (perhaps)
---

These are just some of my thoughts on machined-learned intelligence,
formed by reading a bunch of papers and watching a few talks here and
there.

### Intelligence, tasks, representations and learning

Taking inspiration from Wikipedia, intelligence or intelligent behavior
may be decomposed into these capabilities

1.  Perceive or infer information

2.  Retain that information into knowledge for easy reuse (store and
    factorize experience)

3.  Apply this knowledge towards adaptive and preferred behaviors within
    an environment or context

4.  Ability to measure success and learn from decisions by exploring
    opportunities to improve the steps above

In [Legg2007](https://arxiv.org/pdf/0712.3329.pdf), a quantitative definition of intelligence is proposed,
where the objective (referred to as Universal Intelligence) is
essentially maximized when for each task, the agent tries to minimize
the computational complexity of learning and inference of agent policies
for each task. In essence, to solve each task efficiently.

A **task** is making a decision (or sequence of decisions) based on the
data. Decisions could be classifications, predictions and actions. A
**representation** is a function of the data which is useful for a task.
Usually the input data dimension is very large (pixel space, sampled
waveforms, etc.) and the target space (classifications) are in a much
lower dimensional space. **Learning** is the process of using data (or
experience) to figure out good representations to solve the task.

### What are good representations?

Let $x$ denote the input data, $f(x)$ be a function that maps the input
to a representation $z = f(x)$ which is designed so that the task,
denoted by decision (random variable) $y$ can be solved effectively. The
mapping $f(x)$ is learned via data and tasks. We would like that the
features or representations learned help efficient future task learning
and ideally have the following attributes

1.  Invariant to nuisances

2.  Disentangled, independent, composable, reusable, interpretable

3.  Compressed, minimal, sparse

4.  Hierarchical, multilevel

5.  Task agnostic

6.  Dynamically added via life-long learning

We would like the representation to be invariant to nuisances, such as
translation, rotation, change of scale or lighting, noise, that are not
relevant to predicting $y$

$$
f(x) = f(g \circ x)
$$

Disentangled or independent representations allow composition of factors
of variation to create new unseen concepts (imagination) and allow
abstract reasoning. These factors correspond to finding invariants and
independently transformable aspects of the world. Tasks are defined in
terms of these invariant entities. New concepts can be formed from
logical combination of old concepts. For example, a blue orange.
Disentangled factors of variation can be used for one-shot or zero-shot
learning. New factors can be discovered and added in a life-long
learning setting.

In order to reason about concepts or imagine scenarios, it is essential
to have a representation which is also sparse. We don’t want to reason
about things that are irrelevant to the task at hand. We need to be
really good at discerning irrelevant information and throwing it away.
Sometimes the previously trained representations may need to be adapted
to provide this property.

As we get more abstract and form higher-level concepts, it is necessary
to throw away information and only model things about the objects in the
input that are crucial for the task. This not only makes it more robust,
but also allows reasoning at this higher level of abstraction. It is
likely easier to model the world (physics) at this level as well. We
would like the representation $f(x)$ to have enough information to solve
the task and simultaneously minimize information about the data that is
not relevant for the task $y$. Stated mathematically using the concept
of mutual information and referred to as the _Information Bottleneck_

$$
\min_{f} I(f(x);x) - \lambda I(f(x); y)
$$

We should note that one task’s nuisance is probably another task’s
discriminative data. I’m thinking of a speech recognizer and an
environment (noise) sound classifier, for example. There is some
interesting work [Jacobsen2018](https://arxiv.org/pdf/1802.07088.pdf)
about invertible neural networks where
they show that compression (throwing away information, except at the
last classifier layer) is not required for achieving high classification
performance. Based on this we are speculating representations to have
some sort of hierarchy:

1.  Invertible task-agnostic features: no information loss

2.  Intermediate task group features: information loss via gating,
    masking or attention mechanisms

3.  Task-level features: high-level disentangled abstract latent space

As an example of hierarchical representations, we can have at highest
level (deepest layer) the concept of living organism, then at the next
lower level, the concept of bird and then the level below that the
concept of crow and bluebird and then beaks, wings, legs and so forth.
We could ask, what are similarities between different examples of a
class and these could be color, size, shape differences. As we go deeper
into the hierarchy we lose details about the input that aren’t relevant.
So if we are interested in cat detection, all information about the
background could be thrown away by the time we reach the cat concept.

### Learning and prediction

Deep learning has been successful in various tasks. The model parameters
that are the weights of the layers of the neural network are learned via
backpropagation using stochastic gradient descent (SGD). If we have good
hyperparameters (initial weights and learning rates) and sufficient
training data, SGD can often provide good representations that
generalize. Transfer learning via fine tuning the final layer and/or
representations is a practical method of taking a network trained on one
task (for its representations) and then applying it on another where the
data is not as plentiful. Meta learning addresses the problem of how to
efficiently learn a new task, given experience learning various tasks.
In life-long learning we need to detect shifts in data distribution. We
need to prevent catastrophic forgetting. We need to allocate spare
representation capacity to learn new concepts and share or consolidate
latents where appropriate.

The concept of compute adaptive prediction, in order to effectively
focus on the right information for the task, there may be parts in the
prediction or representation that are not fixed steps and may be
iterative for complicated inputs. This could be a sequential process
driven by RL-based policy. Humans take more time when trying to find an
object that is camouflaged or speech in low SNR, for example. In an
environment with a time aspect, we want to predict multiple outputs for
a single input and learn when the actual observation is divergent from
our predicted future. Having a concept of prediction with constraints
allows to produce outputs that satisfies certain task-dependent
constraints, such as a linguistically correct sentence. We can use
energy/barrier functions for prediction with constraints

$$
\hat{y} = \operatorname*{arg\,min}_y F(x,y)
$$

As an example of learning disentangled factors from data, we have
independent generative factors like position, size, shape, rotation and
color. Then the generated image is an example of entangled data. In
[Higgins2017](https://openreview.net/pdf?id=Sy2fzU9gl) a model
called BetaVAE was proposed to learn the
disentangled representations. The BetaVAE model is able to learn
disentangled representations by modifying the loss function with
additional terms that encourage independence of the latent
representations. A key concept in learning good representations is
constructing objective and loss functions. These representations allow
to traverse across the latent dimensions.

Curriculum learning is a well-designed sequence of tasks that enables
reuse and good representation learning. Open question is how to design
these automatically.

### Final thoughts

For now, we will superficially list a bunch of concepts or buzzwords
that seem to be relevant to the design of intelligent systems, this is
the TL;DR version if at all

- **Gradient-based learning** are learning algorithms that use the
  gradient of the objective function or loss function with respect to
  the differentiable model parameters to nudge the model towards a
  better performance on the task

- **Continual or life-long learning**: is the idea of the model being
  capable of learning to do new tasks and avoid the problem of
  catastrophic forgetting, where the model forgets about tasks it
  learnt in the past. Controlled forgetting is when the model
  purposefully forgets what has been learned in order to learn
  something new in a new situation faster, perhaps a more compact
  factorization of knowledge.

- **Self-supervised learning**: learning representations without the
  labels is termed _unsupervised learning_. Recent work in NLP and
  Vision have shown that good low level features can be learned from
  even a single representative example using augmentation and
  appropriate self-supervised tasks

- **Specialized modeling**: For particular instances in a set of
  tasks, it may be beneficial to route it to a more specialized model
  to provide higher capacity. It helps to understand the environment
  in which we are in to figure out best modeling options for the
  downstream task. Alternately, this could be a special preprocessing
  step on the instance to adapt its domain

- **Memorization**: special case handing for useful concepts and
  exceptions (OOD) to enable faster adaptation when domains switch.
  Clustering and exemplars.

- **Multilevel features**: global vs. local perspective, using
  attention to dynamically focus on different aspects sequentially

- **Metalearning**: figuring out whether transfer learning (final
  stage linear classifier), rapid learning, good initializations,
  adding spare capacity (new representations), decide what
  representations to share, those to modify in a proper way are needed
  to learn new tasks efficiently

- **Disentangled representations**: to enable wider generalizations
  beyond interpolation to nearest example, there needs to be high
  level semantic features that are sparse, independent, disentangled,
  where notions of similarities, new combinations are easily performed

- **Generative modeling**: models learned in the abstract disentangled
  latent space will allow simulation of futures and allow high-level
  action planning. These could be trained via self-supervised
  predictive coding.
