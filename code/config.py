import argparse
import ast
import os.path as osp
import yaml
import random
from easydict import EasyDict as edict
import numpy.random as npr
import torch
from pathlib import Path
from utils import (
    edict_2_dict,
    check_and_create_dir,
    update)
import wandb
import warnings
warnings.filterwarnings("ignore")
# 自定义解析函数，将字符串转换为列表
def parse_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config/base.yaml")
    parser.add_argument("--experiment", type=str, default="conformal_0.5_dist_pixel_100_kernel201")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument('--log_dir', metavar='DIR', default="output_insert")
    # parser.add_argument('--log_dir', metavar='DIR', default="output_src")
    parser.add_argument('--log_dir', metavar='DIR', default="output_change")
    parser.add_argument('--font', type=str, default="none", help="font name")
    parser.add_argument('--semantic_concept', type=str, help="the semantic concept to insert")
    parser.add_argument('--word', type=str, default="none", help="the text to work on")
    parser.add_argument('--prompt_suffix', type=str, default="minimal flat 2d vector. lineal color."
                                                             " trending on artstation. cartoon")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--wandb_user', type=str, default="none")
    parser.add_argument('--image_dir', type=str, default="none")
    
    # Stable Diffusion 2.1 使用与否
    parser.add_argument('--sd_2_1', action='store_true', help='Whether to use Stable Diffusion v2.1')

    # token indices
    parser.add_argument('--token_indices', type=parse_list, default=None, help='Which token indices to alter')

    # 输出路径
    parser.add_argument('--output_path', type=Path, default='./outputs', help='Path to save all outputs to')

    # 去噪步骤数
    parser.add_argument('--n_inference_steps', type=int, default=50, help='Number of denoising steps')

    # 文本引导系数
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Text guidance scale')

    # 应用 attend-and-excite 的最大迭代次数
    parser.add_argument('--max_iter_to_alter', type=int, default=25, help='Number of iterations to apply attend-and-excite')

    # Attention map 分辨率
    parser.add_argument('--attention_res', type=int, default=16, help='Resolution of UNet to compute attention maps over')

    # 是否运行标准的 SD
    parser.add_argument('--run_standard_sd', action='store_true', help='Whether to run standard Stable Diffusion or attend-and-excite')

    # Latent refinement 的阈值
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.5, 0.8], help='Thresholds for latent refinement')

    # 更新去噪 latent 的缩放因子
    parser.add_argument('--scale_factor', type=int, default=20, help='Scale factor for updating the denoised latent z_t')

    # 缩放范围
    parser.add_argument('--scale_range', type=float, nargs=2, default=(1.0, 0.5), help='Scaling range for the scale factor')

    # 是否在计算最大注意力值之前应用高斯平滑
    parser.add_argument('--smooth_attentions', action='store_true', help='Whether to apply Gaussian smoothing before computing maximum attention values')

    # 高斯平滑的标准差
    parser.add_argument('--sigma', type=float, default=0.5, help='Standard deviation for Gaussian smoothing')

    # 高斯平滑的核大小
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for Gaussian smoothing')

    # 是否保存 cross attention map
    parser.add_argument('--save_cross_attention_maps', action='store_true', help='Whether to save cross attention maps')

    # BoxDiff 相关参数
    parser.add_argument('--bbox', type=parse_list, default=None, help='Bounding box coordinates')
    parser.add_argument('--color', type=str, nargs='+', default=['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black'], help='Colors for bounding boxes')
    parser.add_argument('--P', type=float, default=0.2, help='P value for BoxDiff')
    parser.add_argument('--L', type=int, default=1, help='Number of pixels around the corner to be selected')
    parser.add_argument('--refine', action='store_true', help='Whether to refine the result')
    parser.add_argument('--gligen_phrases', type=str, nargs='+', default=['', ''], help='Gligen phrases for BoxDiff')
    parser.add_argument('--n_splits', type=int, default=4, help='Number of splits for BoxDiff')
    parser.add_argument('--which_one', type=int, default=1, help='Which split to use for BoxDiff')
    parser.add_argument('--eval_output_path', type=Path, default='./outputs/eval', help='Evaluation output path')
    
    
    # 权重 骨架loss
    parser.add_argument('--angeles_w', type=float, default=0.5, help='weight of loss')
    
    cfg = edict()
    args = parser.parse_args()
    # 将 argparse 的结果赋值给 cfg
    cfg.sd_2_1 = args.sd_2_1
    cfg.token_indices = args.token_indices
    cfg.output_path = args.output_path
    cfg.n_inference_steps = args.n_inference_steps
    cfg.guidance_scale = args.guidance_scale
    cfg.max_iter_to_alter = args.max_iter_to_alter
    cfg.attention_res = args.attention_res
    cfg.run_standard_sd = args.run_standard_sd
    cfg.thresholds = args.thresholds
    cfg.scale_factor = args.scale_factor
    cfg.scale_range = args.scale_range
    cfg.smooth_attentions = args.smooth_attentions
    cfg.sigma = args.sigma
    cfg.kernel_size = args.kernel_size
    cfg.save_cross_attention_maps = args.save_cross_attention_maps

    # BoxDiff 相关
    cfg.bbox = args.bbox
    cfg.color = args.color
    cfg.P = args.P
    cfg.L = args.L
    cfg.refine = args.refine
    cfg.gligen_phrases = args.gligen_phrases
    cfg.n_splits = args.n_splits
    cfg.which_one = args.which_one
    cfg.eval_output_path = args.eval_output_path
    # with open('TOKEN', 'r') as f:
    
    setattr(args, 'token', "token")
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.font = args.font
    cfg.semantic_concept = args.semantic_concept
    cfg.word = cfg.semantic_concept if args.word == "none" else args.word

    cfg.caption = f"{args.semantic_concept}. {args.prompt_suffix}"
    cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
    # if args.word in cfg.word:
    cfg.word =  args.word
    # else:
    #   raise ValueError(f'letter should be in word')
    cfg.batch_size = args.batch_size
    cfg.token = args.token
    cfg.use_wandb = args.use_wandb
    cfg.wandb_user = args.wandb_user
    cfg.letter = f"{args.word}_scaled"
    cfg.target = f"code/data/init/{cfg.letter}"
    
    cfg.angeles_w = args.angeles_w
    cfg.image_dir = args.image_dir

    return cfg


def set_config():

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg_full = yaml.load(f, Loader=yaml.FullLoader)

    # recursively traverse parent_config pointers in the config dicts
    cfg_key = cfg_arg.experiment
    cfgs = [cfg_arg]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get('parent_config', 'baseline')

    # allowing children configs to override their parents
    cfg = edict()
    for options in reversed(cfgs):
        update(cfg, options)
    del cfgs

    # set experiment dir
    signature = f"{cfg.letter}_seed_{cfg.seed}"
    cfg.experiment_dir = \
        osp.join(cfg.log_dir, cfg.font, signature)
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    print('Config:', cfg)

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(project="Word-As-Image", entity=cfg.wandb_user,
                   config=cfg, name=f"{signature}", id=wandb.util.generate_id())

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
    else:
        assert False

    cfg.loss.conformal.angeles_w = cfg_arg.angeles_w
    cfg.image_dir = cfg_arg.image_dir
    
    return cfg
