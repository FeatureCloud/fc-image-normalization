# Image Normalization FeatureCloud App

## Description
Using the Image Normalization app, users can normalize and/or standardize their local image datasets using different FeatureCloud platform techniques.
This app supports image datasets in NumPy files.

## Input
- train.npy containing the local training data  
- test.npy containing the local test data
`train.npy` and `test.npy` should include same number of samples and the structure should be same as [CrossValidation](../CrossValidation/README.md#input) app.
## Output
- train.npy: Normalized training data 
- test.npy: Normalized test data
Both files have the same structure as the input files.
  
## Methods
- variance:     x<sub>j</sub> = (x<sub>j</sub> - &mu;<sub>j</sub>) / &sigma;<sub>j</sub>

Where
- x: inputs image
- j: channel index
- &mu;<sub>j</sub>: global mean for channel j
- &sigma;<sub>j</sub>: global standard deviation for channel j
  
    

## Workflows
Image Normalization app can be used alongside the following apps in the same workflow:
- Pre: Cross-Validation for Image datasets
- Post: Various Analysis apps that support `.npy` and `.npz` format for their input files (e.g., Deep Learning)

![Workflow](data/images/ImageNormalization.png)
## Config
Following config information should be included in the `config.yml` to run the Image Normalization app in a workflow:
```
fc_image_normalization:
  local_dataset:
    train: train.npy
    test: test.npy
    target_value: same-sep
  method: variance
  logic:
    mode: file
    dir: .
  use_smpc: false
  result:
    train: train.npy
    test: test.npy
```
