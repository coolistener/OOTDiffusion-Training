CUDA_VISIBLE_DEVICES=0
python train_ootd.py \
  --model_type "dc" \
  --train_epochs 50 \
  --batch_size 24 \
  --learning_rate 1e-5 \
  --conditioning_dropout_prob 0.1 \
  --mixed_precision "fp16" \
  --img_height 512 \
  --img_width 384 \
  --num_workers 14 \
  --dataset_dir "../DressCode" \
  --vit_path "../models/clip-vit-large-patch14" \
  --vae_path "../models/stable-diffusion-v1-5/vae" \
  --unet_path "../models/stable-diffusion-v1-5/unet" \
  --tokenizer_path "../models/stable-diffusion-v1-5/tokenizer" \
  --text_encoder_path "../models/stable-diffusion-v1-5/text_encoder" \
  --scheduler_path "../models/stable-diffusion-v1-5/scheduler/scheduler_config.json" \
  --first_epoch 5 \
  --checkpoint_path './train/checkpoints_dc/epoch_4/checkpoint-epoch4.pt'