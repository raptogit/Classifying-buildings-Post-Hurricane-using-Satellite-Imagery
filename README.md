# Classifying-buildings-Post-Hurricane-using-Satellite-Imagery

Damage assessment is crucial for emergency managers to respond quickly and allocate resources after a hurricane. Quantifying the number of flooded/damaged structures, which is generally done via ground survey, is one technique to measure the magnitude of the damage. This procedure can be time-consuming and labor-intensive. We create a convolutional neural network from the ground up and compare it to a widely used neural network for object classification. In a case study of building damage, we demonstrate the promise of our damage annotation approach (almost 97 percent accuracy).The data are satellite images from Texas after Hurricane Harvey divided into two groups (damage and no_damage). The goal is to make a model which can automatically identify if a given region is likely to contain flooding damage.

**We are going on a journey to make it a reality**

## Problem Statement :<br>
### To identify if the particular house is damaged or not from the Satellite Imagery provided. <br>


# Requirements
- [Matplotlib](https://matplotlib.org/) (Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.)
- [Tensorflow](https://www.tensorflow.org/) (The core open source library to help you develop and train ML models.) 
- [Sklearn](https://scikit-learn.org/) (Scikit-learn is a free software machine learning library for Python.)
- [Numpy](https://numpy.org/) (NumPy can be used to perform a wide variety of mathematical operations on arrays.)
- [Numpy](https://numpy.org/) (NumPy can be used to perform a wide variety of mathematical operations on arrays.)

# Table Of Contents
## Steps to take : <br>
<b>Step A</b>: Taking a look at the data and images.<br>
<b>Step B</b>: Data pre-processing.<br>
<b>Step C</b>: Building different models using CNN.<br>
<b>Step D</b>: Applying tranfer learning.<br>
<b>Step E</b>: Comaring results and conclusion.<br>

# In a Nutshell:
- <b>Step A</b>: Taking a look at the data and images.<br>
![Damaged and undamaged houses post hurricane](Classifying-buildings-Post-Hurricane-using-Satellite-Imagery/main/asses_for_github/output.png)
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```






# Contributing
Any kind of enhancement or contribution is welcomed.






