import sys
import os
from shutil import copyfile
import argparse
from utils.dataset import VITONDataset, VITONDataLoader
sys.path.append(r'../ootd')
# models import
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from diffusers.optimization import get_scheduler 
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler, PNDMScheduler
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel

from tqdm import tqdm
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
import math
from pathlib import Path
import logging

#-----args-----
def get_args():
    
    parser = argparse.ArgumentParser()
    
    # training configs
    parser.add_argument("--model_type", type=str, default='hd', help="hd or dc.")
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--first_epoch", type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--learning_rate",type=float,default=5e-5)
    parser.add_argument("--conditioning_dropout_prob",type=float,default=0.1,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["normal","fp16", "bf16"],
        help=(
            "Whether to use mixed precision."
        ),
    )
    
    # dataset configs
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument('--img_width', type=int, default=384)
    parser
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')

    # paths
    parser.add_argument('--dataset_dir', type=str, default='../VITON-HD')
    parser.add_argument("--vit_path", type=str, default="../models/clip-vit-large-patch14")
    parser.add_argument("--vae_path", type=str, default="../models/stable-diffusion-v1-5/vae")
    parser.add_argument("--unet_path", type=str, default="../models/stable-diffusion-v1-5/unet")
    parser.add_argument("--tokenizer_path", type=str, default="../models/stable-diffusion-v1-5/tokenizer")
    parser.add_argument("--text_encoder_path", type=str, default="../models/stable-diffusion-v1-5/text_encoder")
    parser.add_argument("--scheduler_path", type=str, default="../models/stable-diffusion-v1-5/scheduler/scheduler_config.json")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint to resume training.")
    
    
    # lr scheduler configs
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    
    args, unknown = parser.parse_known_args()
    return args

args = get_args()

#-----prepare dataset-----
test_dataset = VITONDataset(args, "test")
test_loader = VITONDataLoader(args, test_dataset)
train_dataset = VITONDataset(args, "train")
train_dataloader = VITONDataLoader(args, train_dataset)
train_dataloader = train_dataloader.data_loader

#-----load models-----
vae = AutoencoderKL.from_pretrained(args.vae_path)

unet_garm = UNetGarm2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)
unet_vton = UNetVton2DConditionModel.from_pretrained(args.unet_path,use_safetensors=True)
# unet_garm = torch.nn.DataParallel(unet_garm)
# unet_vton = torch.nn.DataParallel(unet_vton)

noise_scheduler = PNDMScheduler.from_pretrained(args.scheduler_path)
auto_processor = AutoProcessor.from_pretrained(args.vit_path)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.vit_path)
tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_path)
text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_path)

#-----models configs-----
# unet_vton(denoising UNet)in_channels=4 --> in_channels=8
if unet_vton.conv_in.in_channels == 4:
    with torch.no_grad():
        new_in_channels = 8
        # create a new conv layer with 8 input channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet_vton.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )
        torch.nn.init.kaiming_normal_(conv_new.weight) 
        conv_new.weight.data = conv_new.weight.data * 0.  
        conv_new.weight.data[:, :4] = unet_vton.conv_in.weight.data  
        conv_new.bias.data = unet_vton.conv_in.bias.data  
        unet_vton.conv_in = conv_new  
        print('Add 4 zero-initialized channels to the first convolutional layer of the denoising UNet to support our input with 8 channels')
else:
    print("in_channels = 8")

vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

vae.requires_grad_(False)
unet_garm.requires_grad_(True)
unet_vton.requires_grad_(True)
image_encoder.requires_grad_(False)
text_encoder.requires_grad_(False)

unet_garm.train()
unet_vton.train()

#-----set training environment-----
# run on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# training type
weight_dtype=torch.float32
if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

vae.to(device,dtype=weight_dtype)
unet_garm.to(device)
unet_vton.to(device)
image_encoder.to(device,dtype=weight_dtype)
text_encoder.to(device,dtype=weight_dtype)

#-----training-----
# configs
model_type=args.model_type
batch_size = args.batch_size
train_epochs=args.train_epochs
learning_rate=args.learning_rate
gradient_accumulation_steps=args.gradient_accumulation_steps
first_epoch = args.first_epoch

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training")
loss_log = []
loss_log_file = "./train/loss_log.txt"

# optimizer
all_params = list(unet_garm.parameters()) + list(unet_vton.parameters())
optimizer = torch.optim.AdamW(all_params,lr=learning_rate)

# learning rate scheduler
# lr_scheduler = get_scheduler(
#     args.lr_scheduler,
#     optimizer=optimizer,
#     num_training_steps=train_epochs * len(train_dataloader),
#     num_warmup_steps=args.lr_warmup_steps,
#     num_cycles=args.lr_num_cycles,
#     power=args.lr_power,
#     last_epoch=int(first_epoch) - 1,
# )

# scaler for AMP
scaler = torch.cuda.amp.GradScaler()

# method to tokenize captions
def tokenize_captions(captions, max_length):
    inputs = tokenizer(
        captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# load checkpoint
if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path)
    unet_garm.load_state_dict(checkpoint['unet_garm'])
    unet_vton.load_state_dict(checkpoint['unet_vton'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    first_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resumed training from checkpoint {args.checkpoint_path} at epoch {first_epoch}")

# training loop
logger.info("Start training")
for epoch in tqdm(range(first_epoch, train_epochs)):
    logger.info(f"Epoch {epoch}")
    epoch_loss = 0
    for step, batch in enumerate(train_dataloader):
        logger.info(f"Step {step}")
        
        optimizer.zero_grad()
        
        # get original image data
        image_garm = batch['garm']['paired'].to(device).to(dtype=weight_dtype)
        image_vton = batch['img_agnostic'].to(device).to(dtype=weight_dtype)
        image_ori = batch['img'].to(device).to(dtype=weight_dtype)

        # get garment prompt embeddings
        prompt_image = auto_processor(images=image_garm, return_tensors="pt").data['pixel_values'].to(device)
        prompt_image = image_encoder(prompt_image).image_embeds
        prompt_image = prompt_image.unsqueeze(1)
        
        if model_type == 'hd':
            prompt_embeds = text_encoder(tokenize_captions(['']*batch_size, 2).to(device))[0]
            prompt_embeds[:, 1:] = prompt_image[:]
            
        elif model_type == 'dc':
            prompt_embeds = text_encoder(tokenize_captions(batch['label'], 3))[0]
            prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
        else:
            raise ValueError("model_type must be 'hd' or 'dc'!")
        
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device) 
        bs_embed, seq_len, _ = prompt_embeds.shape
        num_images_per_prompt = 1
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # original image data preprocess
        image_garm = image_processor.preprocess(image_garm)
        image_vton = image_processor.preprocess(image_vton)
        image_ori = image_processor.preprocess(image_ori)

        # get model img latents
        latents = vae.encode(image_ori).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # add noise to latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # get garm and vton img latents
        image_latents_garm = vae.encode(image_garm).latent_dist.mode()
        image_latents_garm = torch.cat([image_latents_garm], dim=0).to(dtype=weight_dtype)

        image_latents_vton = vae.encode(image_vton).latent_dist.mode()
        image_latents_vton = torch.cat([image_latents_vton], dim=0)
        latent_vton_model_input = torch.cat([noisy_latents, image_latents_vton], dim=1).to(dtype=weight_dtype)

        # outfitting dropout
        if args.conditioning_dropout_prob is not None:
            random_p = torch.rand(bsz, device=latents.device)
            image_mask_dtype = image_latents_garm.dtype
            image_mask = 1 - (
                (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            image_latents_garm = image_mask * image_latents_garm
        
        with torch.cuda.amp.autocast():       
            # outfitting fusion
            sample, spatial_attn_outputs = unet_garm(
                image_latents_garm,
                0,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            spatial_attn_inputs = spatial_attn_outputs.copy()

            # outfitting denoising
            noise_pred = unet_vton(
                latent_vton_model_input,
                spatial_attn_inputs,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]  
        
            # calculate loss
            noise_loss= F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = noise_loss
            
            torch.cuda.empty_cache()

        # backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # lr_scheduler.step()
        torch.cuda.empty_cache()
        logger.info(f"Training loss: {loss.item()} at step {step} epoch {epoch}")
        epoch_loss += loss.item()

    epoch_loss /= len(train_dataloader)
    loss_log.append(epoch_loss)
    logger.info(f"Training loss: {epoch_loss} at epoch {epoch}")
    with open(loss_log_file, "a") as f:
        f.write(f"Epoch {epoch}: Loss {epoch_loss}\n")

    # save checkpoints
    if (epoch % 5 == 0 and epoch != 0) or epoch == (args.train_epochs - 1) :
        state_dict_unet_vton = unet_vton.state_dict()
        for key in state_dict_unet_vton.keys():
            state_dict_unet_vton[key] = state_dict_unet_vton[key].to('cpu')
        state_dict_unet_garm = unet_garm.state_dict()
        for key in state_dict_unet_garm.keys():
            state_dict_unet_garm[key] = state_dict_unet_garm[key].to('cpu')
        
        checkpoint_dir = f"./train/checkpoints_{model_type}/epoch_{str(epoch)}"
        checkpoint_dir_vton = f"./train/checkpoints_{model_type}/epoch_{str(epoch)}/unet_vton"
        checkpoint_dir_garm = f"./train/checkpoints_{model_type}/epoch_{str(epoch)}/unet_garm"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir_vton, exist_ok=True)
        os.makedirs(checkpoint_dir_garm, exist_ok=True)
        
        save_file(state_dict_unet_vton, os.path.join(checkpoint_dir_vton, "diffusion_pytorch_model.safetensors"))
        copyfile("./train/unet_configs/unet_vton.json", os.path.join(checkpoint_dir_vton, "config.json"))
        save_file(state_dict_unet_garm, os.path.join(checkpoint_dir_garm, "diffusion_pytorch_model.safetensors"))
        copyfile("./train/unet_configs/unet_garm.json", os.path.join(checkpoint_dir_garm, "config.json"))
        state = {
            'epoch': epoch,
            'unet_garm': unet_garm.state_dict(),
            'unet_vton': unet_vton.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict()
        }
        torch.save(state, os.path.join(checkpoint_dir,f"checkpoint-epoch{str(epoch)}.pt"))
        logger.info('Checkpoints successfully saved')
    
