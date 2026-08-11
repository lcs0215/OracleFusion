import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from diffusers import HunyuanDiTPipeline
from diffusers.models.embeddings import get_2d_rotary_pos_embed

def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)



class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        print("-----------------------------")
        print(cfg.diffusion.model)
        print("-----------------------------")
        
        self.dytpe = torch.float16
        
        self.pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",cache_dir='./', torch_dtype=torch.float16)
        
        self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.prompt_attention_mask,
            self.negative_prompt_attention_mask,
        ) = self.pipe.encode_prompt(
            prompt=self.cfg.caption,
            device=self.device,
            dtype=self.dytpe,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = True,
            negative_prompt="",
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            max_sequence_length=77,
            text_encoder_index=0,
        )
        
        (
            self.prompt_embeds_2,
            self.negative_prompt_embeds_2,
            self.prompt_attention_mask_2,
            self.negative_prompt_attention_mask_2,
        ) = self.pipe.encode_prompt(
            prompt=self.cfg.caption,
            device=self.device,
            dtype=self.dytpe,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            max_sequence_length=256,
            text_encoder_index=1,
        )


    def forward(self, x_aug):
        sds_loss = 0
        batch_size, _, height, width = x_aug.shape
        height = height
        width = width
        
        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, self.cfg.diffusion.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)
            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)
            # eps = torch.cat([eps] * 2, dim = 1)
            # latent_z = torch.cat([latent_z] * 2, dim = 1)
            
            
            
            target_size = [height, width]            
            add_time_ids = list([1024, 1024] + target_size + [0, 0])
            add_time_ids = torch.tensor([add_time_ids], dtype=self.prompt_embeds.dtype)
            add_time_ids = add_time_ids.to(dtype=self.prompt_embeds.dtype, device=self.device).repeat(
            batch_size * 1, 1)
            
            
            style = torch.tensor([0], device=self.device)
            style = style.to(device=self.device).repeat(batch_size * 1)
            
            
            grid_height = height // 8 // self.pipe.transformer.config.patch_size
            grid_width = width // 8 // self.pipe.transformer.config.patch_size
            base_size = 512 // 8 // self.pipe.transformer.config.patch_size
            
            grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
            
            image_rotary_emb = get_2d_rotary_pos_embed(
                self.pipe.transformer.inner_dim // self.pipe.transformer.num_heads, grid_crops_coords, (grid_height, grid_width)
                )
            
            timestep = torch.tensor(timestep, dtype = z_in.dtype, device = z_in.device)
            
            
            prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds])
            prompt_attention_mask = torch.cat([self.negative_prompt_attention_mask, self.prompt_attention_mask])
            prompt_embeds_2 = torch.cat([self.negative_prompt_embeds_2, self.prompt_embeds_2])
            prompt_attention_mask_2 = torch.cat([self.negative_prompt_attention_mask_2, self.prompt_attention_mask_2])
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            style = torch.cat([style] * 2, dim=0)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                noise_pred = self.pipe.transformer(
                    z_in,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_embedding_mask=prompt_attention_mask,
                    encoder_hidden_states_t5=prompt_embeds_2,
                    text_embedding_mask_t5=prompt_attention_mask_2,
                    image_meta_size=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                ).sample.float()
                 
                noise_pred, _ = noise_pred.chunk(2, dim=1)
                eps_t_uncond, eps_t = noise_pred.chunk(2)

            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            timestep = torch.tensor(timestep, dtype = torch.long, device = z_in.device)
            grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
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
        self.device = device
        with torch.no_grad():
            self.angles = []
            self.reset()


    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        points = points.to(self.device)
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




