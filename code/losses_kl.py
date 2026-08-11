import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from diffusers import KolorsPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
#         self.cfg = cfg
        self.device = device
        # self.pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",cache_dir='./', torch_dtype=torch.float16)
        # self.sd_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_auth_token='', cache_dir='/data/LiCaoshuo/')
        self.pipe = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16,variant="fp16", cache_dir='/data/LiCaoshuo/', use_auth_token='').to(self.device)
        self.sdxl_vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",  # 指定只加载VAE
            torch_dtype=torch.float16  # 你可以根据需要选择浮点精度
        ).to(self.device)
        # self.pipe.vae.encoder.scaling_factor=0.18215
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)
        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        # tokenizer and embed text
        text_inputs = self.pipe.tokenizer("fjijds", padding="max_length",
                                         max_length=256,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_inputs['input_ids'].shape[-1],
                                         return_tensors="pt")

#         print("text_inputs: ", text_inputs, uncond_input)
        num_images_per_prompt=1
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(
                        input_ids=text_inputs['input_ids'].to(self.device) ,
                        attention_mask=text_inputs['attention_mask'].to(self.device),
                        position_ids=text_inputs['position_ids'].to(self.device),
                        output_hidden_states=True)
            # print(text_embeddings.hidden_states)
            prompt_embeds = text_embeddings.hidden_states[-2].permute(1, 0, 2).clone()
            pooled_prompt_embeds = text_embeddings.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            # print(prompt_embeds)
            
            output = self.pipe.text_encoder(
                        input_ids=uncond_input['input_ids'].to(self.device) ,
                        attention_mask=uncond_input['attention_mask'].to(self.device),
                        position_ids=uncond_input['position_ids'].to(self.device),
                        output_hidden_states=True)
            negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
            negative_pooled_prompt_embeds = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
        self.text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        self.text_embeddings = self.text_embeddings.repeat_interleave(1, 0)
        # print(self.text_embeddings.shape)
        

        self.pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        # print(self.pooled_prompt_embeds.shape)
        
        del self.pipe.tokenizer
        del self.pipe.text_encoder


    def forward(self, x_aug):
        sds_loss = 0
        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = self.sdxl_vae.encode(x)
            init_latent_z = init_latent_z.latent_dist
            # print(init_latent_z)
            init_latent_z = (init_latent_z.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, 1000) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            t = torch.cat([timestep] * 2)
            

            time_ids=list((1024, 1024)+ (0,0) + (1024,1024))
            passed_add_embed_dim = (
                self.pipe.unet.config.addition_time_embed_dim * len(time_ids) + 4096
                )
            expected_add_embed_dim = self.pipe.unet.add_embedding.linear_1.in_features

            if expected_add_embed_dim != passed_add_embed_dim:
                raise ValueError(
                    f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
                )
            time_ids=torch.tensor([time_ids], dtype = torch.float16).to(self.device)
            time_ids=torch.cat([time_ids, time_ids], dim=0)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = self.pipe.unet(
                    z_in, 
                    timestep, 
                    encoder_hidden_states=self.text_embeddings,
                    added_cond_kwargs={
                        "text_embeds": self.pooled_prompt_embeds,  # 文本嵌入
                        "time_ids":  time_ids # 时间步嵌入
                    },
                    return_dict=False,
                )[0].float()
            eps_t_uncond, eps_t = result.chunk(2)
            # print(f'eps_t_uncond: {eps_t_uncond}, eps_t: {eps_t}')
            eps_t = eps_t_uncond + 100 * (eps_t - eps_t_uncond)
            
            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
#             print(grad_z)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return 0,sds_loss



class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur,
                                                                        cfg.loss.tone.pixel_dist_kernel_blur), sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)


    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)
            

class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()


    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    
    def get_letter_inds(self, letter_to_insert):
        # for group, l in zip(self.shape_groups, self.target_letter):
            # if l == letter_to_insert:
                # letter_inds = self.shape_groups.shape_ids
                # return letter_inds[0], letter_inds[-1], len(letter_inds)
        # 直接获取第一个形状组
        group = self.shape_groups[0]
        # 确保传入的字母与目标字母匹配
        if self.target_letter == letter_to_insert:
            letter_inds = group.shape_ids
            return letter_inds[0], letter_inds[-1], len(letter_inds)
        else:
            raise ValueError(f"Letter {letter_to_insert} not found in target_letter")

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        # for j, c in enumerate(self.target_letter):
        points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
        start_ind, end_ind, shapes_per_letter = self.get_letter_inds(self.target_letter)
        print(self.target_letter, start_ind, end_ind)
        holes = []
        if shapes_per_letter > 1:
            holes = points_np[start_ind+1:end_ind]
        poly = Polygon(points_np[start_ind], holes=holes)
        poly = poly.buffer(0)
        points_np = np.concatenate(points_np)
        faces = Delaunay(points_np).simplices
        is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
        faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles




