---
title: Predictions on real data and model quantization (Tutorial)
date: 2020-08-28
subject:
  [
    "ml",
    "tutorial",
    "pytorch",
    "audio",
    "speech",
    "kws",
    "voice",
    "cnn",
    "quantization",
  ]
author: RSP
description: Model quantization for speech commands dataset and evaluation on real data
featimg: ./output_files/output_17_0.png
---

In this post, we want to see if the speech commands recognition model we trained in the [previous post](https://jumpml.com/tutorial-pytorch-speechcommands/output/) actually works on real recorded data.

In addition we will learn how to quantize weights in PyTorch. This will help make the model smaller and potentially faster for prediction. Performance may or may not get worse.

This notebook is available on github at this [link](https://github.com/jumpml/pytorch-tutorials/blob/master/SpeechCommands_CNN_quantize.ipynb).

## Model Loading

In a previous post, we trained a simple CNN model to recognize speech commands. We will load the same exact model with the same input size, kernel sizes, layers, etc. There are two steps in model loading

1. Model instantiation: this is the skeleton with space for the parameters
2. Parameter loading from a previously saved .pt file

```python
PATH = "./models/speech_commands_model.pt"
nnModel = models.SpeechCommandsModel().to(device)       # Instantiate our model and move model to GPU if available
nnModel.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
nnModel.eval()
```

    SpeechCommandsModel(
      (conv1): Conv2d(1, 32, kernel_size=(8, 20), stride=(1, 1))
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(32, 8, kernel_size=(4, 10), stride=(1, 1))
      (bn2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=1536, out_features=128, bias=True)
      (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc2): Linear(in_features=128, out_features=11, bias=True)
    )

## Model Quantization

As of PyTorch 1.6.0, there are three ways to quantize a model

1. Dynamic Quantization: quantized weights applied post-training
2. Static Quantization: fuses layers like BatchNorm and Relu, uses data for calibration applied post-training
3. Quantization-aware training: static quantization during training

We will try out the easiest one, dynamic quantization, which I am just going to call weight quantization. Basically we need to tell the quantizer which layers we want to quantize, for e.g. nn.Linear usually has a lot of parameters and is a prime candidate for weight quantization and what the target data type is. Since parameters are float32, the two options available are float16 or qint8. We will convert Conv2d and Linear layers to 8-bit parameters.

```python
quantized_model = torch.quantization.quantize_dynamic(
    nnModel, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
```

```python
utils.print_size_of_model(nnModel)
utils.print_size_of_model(quantized_model)
```

    Size (MB): 0.864021
    Size (MB): 0.271357

```python
lossFn = F.nll_loss  #When we combine nll_loss and log_softmax we get a cross entropy loss
evalSC = eval.evalModel(nnModel, lossFn, device)
evalSCq = eval.evalModel(quantized_model, lossFn, device)
evalSC.evalClass(sc_data.test_loader)
```

    Avg. loss: 0.0007, Accuracy: 94.42631530761719 %  Elapsed Time=126.23865580558777

```python
evalSCq.evalClass(sc_data.test_loader)
```

    Avg. loss: 0.0007, Accuracy: 94.39006805419922 %  Elapsed Time=36.33142900466919

## To Quantize or Not to Quantize?

Here is the deal

1. Size went down from 0.86 MB to 0.27 MB
2. Accuracy is slightly worse, but pretty much the same: 94.42% to 94.39%
3. Processing time of the test set evaluation went from 2 minutes to half a minute!

Usually there are no free lunches, but what we have here is a free lunch, a proverbial no-brainer.

## Recording real-world data

I used Audacity, a free audio editing tool, to record me saying twenty commands one after the other. After that, I wrote up an Energy-based Voice Activity Detector (VAD) which basically finds the words in the long recording:

![](output_files/VAD.png)

At the end, we have a set of twenty files corresponding to the twenty commands I read. The filename contains the true command label.

The next steps are to setup a feature preprocessing pipeline so we can feed the features to the model. After some copy and paste of the dataset code, we create a function:

```python
(X,y) = scd.get_file_features(file, padLR=False)
```

which takes a filename and returns the features X (101 frames of 64 log Mel features) and label y. This function also allows to pad silence to the right of the command or pad on both left and right of the command.

```python
for file in testFiles:
    (X,y) = scd.get_file_features(file, padLR=False)
    (pred, conf)=evalSCq.predictClass(X)
    if pred == y:
        print(f'\033[1;30;47m Ground Truth = {y} \tPrediction = {pred}   \tConfidence={conf * 100 :.2f}%')
    else:
        print(f'\033[1;30;41m Ground Truth = {y} \tPrediction = {pred}   \tConfidence={conf * 100 :.2f}%')

```

    [1;30;47m Ground Truth = 4 	Prediction = 4   	Confidence=69.92%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=91.55%
    [1;30;47m Ground Truth = 8 	Prediction = 8   	Confidence=99.97%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=77.36%
    [1;30;47m Ground Truth = 3 	Prediction = 3   	Confidence=56.72%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=98.91%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=98.08%
    [1;30;47m Ground Truth = 1 	Prediction = 1   	Confidence=95.63%
    [1;30;47m Ground Truth = 7 	Prediction = 7   	Confidence=94.43%
    [1;30;47m Ground Truth = 9 	Prediction = 9   	Confidence=99.46%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=84.82%
    [1;30;47m Ground Truth = 0 	Prediction = 0   	Confidence=99.06%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=75.87%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=95.41%
    [1;30;47m Ground Truth = 5 	Prediction = 5   	Confidence=89.63%
    [1;30;47m Ground Truth = 6 	Prediction = 6   	Confidence=84.86%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=98.45%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=96.55%
    [1;30;41m Ground Truth = 2 	Prediction = 10   	Confidence=44.08%   ===> OH NOOOOOOOOOOOOOO
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=69.66%

```python
# VISUALIZE SOME EXAMPLES
def visualize_file_prediction(file, evalSC, padLR=False):
    (X,y) = scd.get_file_features(file,padLR=padLR)
    (pred, conf)=evalSC.predictClass(X)
    fig=plt.figure(figsize=(8, 12), dpi=80)
    plt.tight_layout()
    plt.imshow(torch.squeeze(X), cmap='jet', origin='lower')
    plt.title(f'Ground Truth: {scd.KNOWN_COMMANDS[int(y)]}    Prediction:{scd.KNOWN_COMMANDS[pred]}')

visualize_file_prediction(testFiles[-2], evalSCq)
```

![png](output_files/output_15_0.png)

What if the speech was more centered? Would the prediction change? Let see.

```python
visualize_file_prediction(testFiles[-2], evalSCq, padLR=True)
```

![png](output_files/output_17_0.png)

```python
for file in testFiles:
    (X,y) = scd.get_file_features(file, padLR=True)
    (pred, conf)=evalSCq.predictClass(X)
    if pred == y:
        print(f'\033[1;30;47m Ground Truth = {y} \tPrediction = {pred}   \tConfidence={conf * 100 :.2f}%')
    else:
        print(f'\033[1;30;41m Ground Truth = {y} \tPrediction = {pred}   \tConfidence={conf * 100 :.2f}%')

```

    [1;30;47m Ground Truth = 4 	Prediction = 4   	Confidence=99.95%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.91%
    [1;30;47m Ground Truth = 8 	Prediction = 8   	Confidence=100.00%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=83.48%
    [1;30;47m Ground Truth = 3 	Prediction = 3   	Confidence=99.73%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.49%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.83%
    [1;30;47m Ground Truth = 1 	Prediction = 1   	Confidence=98.30%
    [1;30;47m Ground Truth = 7 	Prediction = 7   	Confidence=99.97%
    [1;30;47m Ground Truth = 9 	Prediction = 9   	Confidence=100.00%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.73%
    [1;30;47m Ground Truth = 0 	Prediction = 0   	Confidence=100.00%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.26%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.59%
    [1;30;47m Ground Truth = 5 	Prediction = 5   	Confidence=99.99%
    [1;30;47m Ground Truth = 6 	Prediction = 6   	Confidence=99.69%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=97.21%
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=99.59%
    [1;30;47m Ground Truth = 2 	Prediction = 2   	Confidence=98.81%   ==> ALRIGHTY THEN
    [1;30;47m Ground Truth = 10 	Prediction = 10   	Confidence=56.80%

## Conclusion

Quantizing to 8-bit parameters is a great thing to do for our pre-trained speech commands model.

We also noticed the fragility of the model to where in the spectrogram the word is. This may not be an issue in practice, as the model may be run multiple times a second on overlapping data. Yet, it shows that there is more work to do on getting a more invariant representation. We see the need for more tools to perform sensitivity analysis to noise and perturbations like time shift. And the need to understand why the model is getting confused by using model interpretability libraries like [Captum](https://captum.ai). These will be topics for a future post.
