import torch
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionInpaintPipeline
# from custom_pipeline import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
import torch.nn as nn

torch_device = 'cuda:2'
model_path = "sd-fillImgcondT2I-model"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-2000/unet", torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16)
pipe.to(torch_device)


conditioning_images = Image.open('fill50k/conditioning_images/1.png').convert('L')
cond_transforms = transforms.Compose(
        [
            transforms.Resize(512 // 8, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
conditioning_latents = cond_transforms(conditioning_images).to(torch_device).half().unsqueeze(0)
prompt=["pale golden rod circle with old lace background"]



height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 50           # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(42)    # Seed generator to create the inital latent noise

batch_size = len(prompt)


# 7. get the text_embeddings for the passed prompt
text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]


uncond_input = pipe.tokenizer(
    [""] * batch_size, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]


# 9. concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


# 10. generate the initial random noise
latents = torch.randn(
    (batch_size, 4, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device).half()

# 11. initialize the scheduler with our chosen num_inference_steps.
pipe.scheduler.set_timesteps(num_inference_steps)

# 12. The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here:
latents = latents * pipe.scheduler.init_noise_sigma


# 13. write the denoising loop
from tqdm.auto import tqdm
conditioning_latents = torch.cat([conditioning_latents] * 2)

for t in tqdm(pipe.scheduler.timesteps):

    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    # latent_model_input = latents

    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep=t)
    latent_model_input = torch.cat((latent_model_input, conditioning_latents), dim=1)
    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# 14. use the vae to decode the generated latents back into the image
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = pipe.vae.decode(latents).sample

# 15. convert the image to PIL so we can display or save it
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("fill.png")

