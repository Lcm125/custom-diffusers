# custom-diffusers
Provide customized diffusers training and inference code for different needs

You don't need to edit the code from diffusers source! These scripts can directly run once installed diffusers!

## Requirements

diffusers == 0.25.0
torch >= 2.0

## Func List

1. __Image_guided_structure-Text_guided_content__: 

The input and function is similar to controlnet and depth-to-image, but in a mobile way compared to controlnet, while the training script of depth-to-image is not given by diffusers.

2. __Image_guided_structure-Image_guided_content__:

To be completed...