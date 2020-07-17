# Introduction

Flower classifier using ResNet

# Setup

## Requirements

Install tensorflow and other requirements in a virtual env:
```
$ pip install -r requirements.txt
```

## Flower images

Download http://download.tensorflow.org/example_images/flower_photos.tgz and keep the folders `daisy` and `roses` in the `flower_photos` directory.

# Instructions

First, build the dataset:

```
$ python build_dataset.py
building 'training' split
building 'validation' split
building 'testing' split
$ 
```

Then, train the model:

```
$ python train_model.py
```

Each epoch takes ~50 seconds.

# Current results

| label | precision | recall | f1-score  |
|-------|-----------|--------|-----------|
| daisy | 0.86      | 0.85   | 0.85      |
| roses | 0.86      | 0.87   | 0.86      |
