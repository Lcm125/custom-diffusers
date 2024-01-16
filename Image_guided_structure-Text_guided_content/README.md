# Image_guided_structure-Text_guided_content

I use fusing/fill50k dataset for example: https://huggingface.co/datasets/fusing/fill50k

The image and text are normally used in SD, but the contour images are additional used to control the location of circle.

You can find explanation for this code in Chinese here: https://blog.csdn.net/Lizhi_Tech/article/details/134033499 

## Train

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16"  train.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --dataset_name=fusing/fill50k \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-fillImgcondT2I-model"

## Test

python inference.py