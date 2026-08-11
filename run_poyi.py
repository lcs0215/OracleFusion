import subprocess
import random
import json, os
import pandas as pd
import freetype
from PIL import Image
import sys
from poyi import get_poyi_results as get_poyi_results


# 配置参数
USE_WANDB = 0
WANDB_USER = "none"

font_name = "甲骨文字体库"
p = "0.2"
L = "1"

if len(sys.argv) > 1:
    arg_value = sys.argv[1]
else:
    arg_value = "0"
    
gpu_index = int(arg_value)

os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_index}'

def run(CONCEPT, WORD, token_indices, bbox):
    
    SEED = 2563
    angeles_w = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    for _ in range(2):
        SEED = random.randint(1000, 9999)
        i = gpu_index
        # 构造命令行参数
        EXPERIMENT = f"conformal_0.5_dist_pixel_100_kernel201"
        args = [
            "python", "code/main.py",
            "--token_indices", str(token_indices),
            "--bbox", str(bbox),
            "--P", p,
            "--L", L,
            "--experiment", EXPERIMENT,
            "--seed", f"{SEED}",
            "--font", font_name,
            "--use_wandb", str(USE_WANDB),
            "--wandb_user", WANDB_USER,
            "--semantic_concept", CONCEPT,
            "--word", WORD,
            "--log_dir", f"{output_file}",
            "--angeles_w", "0.5",
            "--image_dir", "GT/dataset"
        ]

        subprocess.run(args)
    
cnt=0

output_file = "outputs_poyi3"

os.makedirs(output_file, exist_ok = True)

# for name in list(concept)[100 * gpu_index: 100 * gpu_index + 100]:

concept = get_poyi_results()

for name in list(concept):

    name = name
    bbox = concept[name][1]
    
    if name in concept:
        code = name
        CONCEPT = ''
        token_indices, bbox_list = [], []
        idx = 0
        concept_list = concept[name][0]
        box_guide_tokens = []
        
        for k, x in enumerate(concept_list):
            text = x
            text = text.replace("'", '')
            text=text.replace('"', '')
            text=text.replace(",", '')
            text=text.replace("+", ' and ')
            text=text.replace("(", '')
            text=text.replace(")", '')
            text=text.replace(":", '')
            words = text.split(' ')
            CONCEPT = CONCEPT + text
            
            if k != len(concept_list)-1:
                CONCEPT = CONCEPT + ', '
                box_guide_tokens.append(text + ",")
                bbox_list.append(bbox[k])
            # print(words)
            if k == len(concept_list) - 1:
                continue

            # for i,word in enumerate(words):
            #     token_indices.append(idx+1)
            #     bbox_list.append(bbox[k])
            #     idx+=1
            # idx+=1
        
        # 单独测试
        # token_indices = [i for i in range(1,10)] + [11]
        # bbox_list = [bbox[0]] * 9 + [bbox[1]]
        
        
        # # 不使用box
        # CONCEPT = concept_list[-1]
        # token_indices, bbox_list = None, None
        if cnt < 300:
            run(CONCEPT, code, box_guide_tokens, bbox_list)
        # print(code, cnt)
        cnt+=1
        # print(token_indices)
        # print(bbox_list)
        # print(CONCEPT)
