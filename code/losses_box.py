import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from torch.nn import functional as F
from gaussian_smoothing import GaussianSmoothing

from diffusers import StableDiffusionPipeline
from ptp_utils import AttentionStore, aggregate_attention
import ptp_utils
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

class SDSLoss(nn.Module):
    def __init__(self, cfg, device, control):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained("/home/tione/notebook/lcs/stable-diffusion-v1-5",
                                                       torch_dtype=torch.float16, use_auth_token=cfg.token, safety_checker=None)
        self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.text_embeddings = None
        self.embed_text()
        self.attention_store = control
        ptp_utils.register_attention_control(self.pipe, self.attention_store)

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(self.cfg.caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.cfg.batch_size, 0)
        # del self.pipe.tokenizer
        # del self.pipe.text_encoder

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False,
                                         bbox: List[int] = None,
                                         config=None,
                                         ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        cnt = 0
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]

            box = [max(round(b / (512 / image.shape[0])), 0) for b in bbox[cnt]]
            x1, y1, x2, y2 = box
            cnt += 1

            # coordinates to masks
            obj_mask = torch.zeros_like(image)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(obj_mask.device)
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            k = (obj_mask.sum() * config.P).long()
            max_indices_list_fg.append((image * obj_mask).reshape(-1).topk(k)[0].mean())

            # Outer-Box constraint
            k = (bg_mask.sum() * config.P).long()
            max_indices_list_bg.append((image * bg_mask).reshape(-1).topk(k)[0].mean())

            # # Corner Constraint
            # gt_proj_x = torch.max(obj_mask, dim=0)[0]
            # gt_proj_y = torch.max(obj_mask, dim=1)[0]
            # corner_mask_x = torch.zeros_like(gt_proj_x)
            # corner_mask_y = torch.zeros_like(gt_proj_y)

            # # create gt according to the number config.L
            # N = gt_proj_x.shape[0]
            # corner_mask_x[max(box[0] - config.L, 0): min(box[0] + config.L + 1, N)] = 1.
            # corner_mask_x[max(box[2] - config.L, 0): min(box[2] + config.L + 1, N)] = 1.
            # corner_mask_y[max(box[1] - config.L, 0): min(box[1] + config.L + 1, N)] = 1.
            # corner_mask_y[max(box[3] - config.L, 0): min(box[3] + config.L + 1, N)] = 1.
            # dist_x.append((F.l1_loss(image.max(dim=0)[0], gt_proj_x, reduction='none') * corner_mask_x).mean())
            # dist_y.append((F.l1_loss(image.max(dim=1)[0], gt_proj_y, reduction='none') * corner_mask_y).mean())

        # return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y
        return max_indices_list_fg, max_indices_list_bg
    
    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   bbox: List[int] = None,
                                                   cfg=None,
                                                   ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        # max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
        #     attention_maps=attention_maps,
        #     indices_to_alter=indices_to_alter,
        #     smooth_attentions=smooth_attentions,
        #     sigma=sigma,
        #     kernel_size=kernel_size,
        #     normalize_eot=normalize_eot,
        #     bbox=bbox,
        #     config=cfg,
        # )
        # return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y
        max_attention_per_index_fg, max_attention_per_index_bg = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bbox=bbox,
            config=cfg,
        )
        return max_attention_per_index_fg, max_attention_per_index_bg
    
    def _compute_loss(self,max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss
        
    def forward(self, x_aug):
        sds_loss = 0

        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents
        
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
        if self.cfg.bbox != None:
            
            # compute box loss
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
                noise_pred_text = self.pipe.unet(
                    z_in, 
                    timestep, 
                    encoder_hidden_states=self.text_embeddings).sample
            self.pipe.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index_fg, max_attention_per_index_bg = self._aggregate_and_get_max_attention_per_token(
                attention_store=self.attention_store,
                indices_to_alter=self.cfg.token_indices,
                attention_res=self.cfg.attention_res,
                smooth_attentions=self.cfg.smooth_attentions,
                sigma=self.cfg.sigma,
                kernel_size=self.cfg.kernel_size,
                normalize_eot=self.cfg.sd_2_1,
                bbox=self.cfg.bbox,
                cfg=self.cfg,
                )
            
            _, box_loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, [], [])
            
            with torch.inference_mode():
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    eps_t_uncond, eps_t = noise_pred_text.float().chunk(2)

                eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (eps_t - eps_t_uncond)

                # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
                grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
                assert torch.isfinite(grad_z).all()
                grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

            sds_loss = grad_z.clone() * latent_z
            del grad_z

            sds_loss = sds_loss.sum(1).mean()
            
            return box_loss * self.weight, sds_loss, timestep[0].item()
            
        else:
            box_loss=0
        with torch.inference_mode():    

            # compute sds loss
            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return box_loss * self.weight, sds_loss
    
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
        self.device = device
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
        for i in range(len(self.faces)):
            triangles = points.to(self.device)[self.faces[i].to(self.device)]
            triangles_roll_a = points.to(self.device)[self.faces_roll_a[i].to(self.device)]
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
        is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces])
        faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles




