from typing import Mapping
import os
import numpy as np
from PIL import Image
from drawer import draw_rectangle, DashedImageDraw, draw_bboxes
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses_box import SDSLoss, ToneLoss, ConformalLoss
from config import set_config
from ptp_utils import AttentionStore
from vis_utils import show_cross_attention
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word,
    create_video)
import wandb
import warnings

######################################
from scipy.spatial import Delaunay
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class SkeletonLoss:
    def __init__(self, 
                 parameters: EasyDict, 
                 device: torch.device, 
                 sk_points):
        self.device = device
        self.parameters = parameters
        self.sk_points = sk_points
        
        self.faces, self.direct_vector = self.init_faces(device)
        
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.norm_vectors = []
            self.reset()

        
    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[i].to(self.device)[self.faces[i].to(self.device)]
            triangles_roll_a = points[i].to(self.device)[self.faces_roll_a[i].to(self.device)]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)

        return angles_
    
    def get_norm_vectors(self, points: torch.Tensor) -> torch.Tensor:
        norm_vectors = []
        for i in range(len(self.direct_vector)):
            item_vector_index = self.direct_vector[i]
            item_points = points[i]
            
            vectors = item_points[item_vector_index]
            vectors = vectors[:, 0] - vectors[:, 1]
            
            norm = torch.norm(vectors, dim=-1, p = 2)
            # 计算法向量
            vectors = vectors / norm[:, None]
            
            norm_vectors.append(vectors)

        return norm_vectors

    def reset(self):
        points = [point.clone().detach().to(self.device) for point in self.parameters.point]
        sk_points = [
            torch.from_numpy(sK_point).to(self.device, dtype=torch.float32)
                for sK_point in self.sk_points]
        
        points = [torch.cat([point, sk_point[:, 0, :]], dim = 0)
                  for point, sk_point in zip(points, sk_points)]

        self.angles = self.get_angles(points)
        
        self.norm_vectors = self.get_norm_vectors(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        vector_group = []
        for sk_point, point in zip(self.sk_points, self.parameters.point):
            points_np = np.vstack([point.clone().detach().cpu().numpy(), sk_point[:, 0, :]])
            
            
            sk_start_index = point.shape[0]
            length = points_np.shape[0]
            sk_indexs = list(range(sk_start_index, length))
            
            
            faces = Delaunay(points_np).simplices
            holes = []
            poly = Polygon(points_np, holes=holes)
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces])
            faces = faces[is_intersect]
            vectors = []
            
            for face in faces:        
                for i, sim_index in enumerate(face):
                    if sim_index in sk_indexs:
                        if face[(i + 1) % 3] not in sk_indexs:
                            vectors.append([sim_index, face[(i + 1) % 3]])
                        if face[(i + 2) % 3] not in sk_indexs:
                            vectors.append([sim_index, face[(i + 2) % 3]])

            # 去重复
            vectors_ = [f"{item[0]},{item[1]}" for item in vectors]
            vectors_ = set(vectors_)
            vectors = [[int(num) for num in item.split(",")] for item in vectors_]

            # 获得对应的 向量 以及 对应的 三角形的端点的index
            vector_group.append(torch.tensor(vectors).to(device, dtype=torch.int64))            
            faces_.append(torch.from_numpy(faces).to(device, dtype=torch.int64))
        
        return faces_, vector_group


    def get_angle_loss(self, points) -> torch.Tensor:
        loss_angles = 0
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles

    
    def get_dt_nrom_angle_loss(self, points) -> torch.Tensor:
        angle_dt_losses = 0
        norm_vectors = self.get_norm_vectors(points)
        for i in range(len(self.norm_vectors)):
            src_norm_vectors = self.norm_vectors[i]
            target_norm_vectors = norm_vectors[i]
            
            # 计算两组法向量的夹角
            dot_product = torch.sum(src_norm_vectors * target_norm_vectors, dim=-1)
            norm = torch.norm(src_norm_vectors, dim=-1, p = 2) * torch.norm(target_norm_vectors, dim=-1, p = 2)
            angle_dt = dot_product / norm
            
            angle_dt_loss = torch.mean(nnf.relu(-angle_dt))
            angle_dt_losses += angle_dt_loss
        
        return angle_dt_losses

    def __call__(self, weights: torch.Tensor = torch.tensor([1.0, 1.0])) -> torch.Tensor:
        points = [point.to(self.device) for point in self.parameters.point]
        sk_points = [
            torch.from_numpy(sK_point).to(self.device, dtype=torch.float32)
                for sK_point in self.sk_points]
        
        points = [torch.cat([point, sk_point[:, 0, :]], dim = 0)
                  for point, sk_point in zip(points, sk_points)]
        
        # return self.get_dt_nrom_angle_loss(points) + self.get_angle_loss(points) * 0.5
        # return self.get_dt_nrom_angle_loss(points) * weights[0] + self.get_angle_loss(points) * weights[1]
        return self.get_angle_loss(points)
    

######################################
def get_word_indices(word_list, tokenizer):
    word_indices = []
    token_pointer = 1  # 跳过 [CLS] 或其他起始标记
    for word in word_list:
        word_len = len(tokenizer.encode(word, add_special_tokens=False))
        indices = list(range(token_pointer, token_pointer + word_len))
        word_indices.append(indices)
        token_pointer += word_len
    return word_indices



warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0

colors = ["red", "green", "blue", "yellow", "purple"]
def init_shapes(device, svg_path, trainable: Mapping[str, bool]):

    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters



if __name__ == "__main__":
    cfg = set_config()
    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()
    controller = AttentionStore()
    print("preprocessing")
    print(cfg.bbox)
    
    
    
    
    scaled_ratio = 512/600
    if cfg.bbox is not None:
        # 根据bbox 进行预处理
        from transformers import CLIPTokenizer
        
        tokenizer = CLIPTokenizer.from_pretrained('/home/tione/notebook/lcs/stable-diffusion-v1-5/tokenizer')
        token_indices = get_word_indices(cfg.token_indices, tokenizer)
        bbox = []
        token_indices_new = []
        for i in range(len(token_indices)):
            if cfg.token_indices[i][-1] == ",":
                token_indices[i] = token_indices[i][:-1]
                
            bbox += ([cfg.bbox[i]] * len(token_indices[i]))
            token_indices_new += token_indices[i]
        
        
        cfg.token_indices = token_indices_new
        cfg.bbox = bbox

        cfg.bbox, obc_utils = preprocess(cfg.font, cfg.word, cfg.bbox, cfg.image_dir)
        cfg.bbox = (cfg.bbox * scaled_ratio).tolist()
    else:
        _, obc_utils = preprocess(cfg.font, cfg.word, cfg.bbox, cfg.image_dir)
    
    print(cfg.bbox)
    if cfg.loss.use_sds_loss:
        sds_loss = SDSLoss(cfg, device, controller)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(device, svg_path=cfg.target, trainable=cfg.trainable)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]
    pydiffvg.imwrite(img_init.detach().cpu(), 'rendered_beziers.png', gamma=2.2)
    
    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.loss.conformal.use_conformal_loss:
        # conformal_loss = ConformalLoss(parameters, device, "", shape_groups)
        
        conformal_loss = SkeletonLoss(parameters, device, obc_utils.reduce_sk_points(obc_utils.resize_sk_points, 1))

    lr_lambda = lambda step: learning_rate_decay(step, cfg.lr.lr_init, cfg.lr.lr_final, num_iter,
                                                 lr_delay_steps=cfg.lr.lr_delay_steps,
                                                 lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]['lr']}, step=step)
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        image_dir = os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png")
        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            save_image(img, image_dir, gamma)
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
            check_and_create_dir(filename)
            save_svg.save_svg(
                filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()
        
        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = data_augs.forward(x)

        if step == 0:
            image_dir

        # 分布变成 -1,1
        x_aug = (x_aug - 0.5) * 2
        # compute diffusion loss per pixel
        box_loss, loss, timestep = sds_loss(x_aug)
        
        
        loss += box_loss
        if cfg.use_wandb:
            wandb.log({"sds_loss": loss.item()}, step=step)

        if cfg.loss.tone.use_tone_loss:
            tone_loss_res = tone_loss(x, step)
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_loss_res}, step=step)
            loss = loss + tone_loss_res

        if cfg.loss.conformal.use_conformal_loss:
            loss_angles = conformal_loss()
            loss_angles = cfg.loss.conformal.angeles_w * loss_angles
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss = loss + loss_angles

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()
        
        
        ### draw attention map
        if cfg.bbox != None and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            heat_map_item_path = os.path.join(cfg.experiment_dir, f'attention_map_items', f"{step}_{timestep}")
            
            os.makedirs(os.path.join(cfg.experiment_dir, f'attention_map'), exist_ok = True)
            os.makedirs(heat_map_item_path, exist_ok = True)
            
            org_img = x_aug[0].detach().permute([1, 2, 0]).cpu().numpy()
            org_img = ((org_img * 0.5 + 0.5) * 255).astype(np.uint8)
            with torch.no_grad():
                figure_show, heatmaps = show_cross_attention(
                    cfg.caption,
                    controller,
                    sds_loss.pipe.tokenizer,
                    cfg.token_indices,
                    16,
                    from_where = ["down"], orig_image = Image.fromarray(org_img),
                )
                figure_show.save(os.path.join(cfg.experiment_dir, f'attention_map/{step}_{timestep}_attention.png'))

                for index, heatmap in enumerate(heatmaps):
                    heatmap.save(os.path.join(heat_map_item_path, f'{index}.png'))

    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)
    
    
    
    if cfg.bbox != None:
        cfg.bbox = (np.array(cfg.bbox) * (600/512)).tolist()
        canvas = Image.fromarray(np.zeros((600, 600, 3), dtype=np.uint8) + 220)
        draw = DashedImageDraw(canvas)
        k =0
        for i, box in enumerate(cfg.bbox):
            x1, y1, x2, y2 = box
            draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=cfg.color[k], width=5)
            if i > 0 and cfg.token_indices[i] != cfg.token_indices[i-1]+1:
                k+=1
        canvas.save(cfg.experiment_dir  + '/canvas.png')
        draw_bboxes(os.path.join(cfg.experiment_dir, 'video-png/iter0000.png'), cfg.bbox, colors, os.path.join(cfg.experiment_dir, f'{cfg.word}_box.png'))
    
    if cfg.save.image:
        filename = os.path.join(
            cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()
