import os
import pandas as pd
import json
import uuid
from PIL import Image
import os
import re
import numpy as np
import cv2

from swift.tuners import Swift
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch
def draw(img_path, bboxes):
    img = cv2.imread(img_path)
    h,w,_ =img.shape
#     img = cv2.resize(img,(1000,1000))
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        print(x_min, y_min, x_max, y_max)
        cv2.rectangle(img, (int(w * x_min/1000), int(h * y_min/1000)), (int(w * x_max/1000), int(h*y_max/1000)), (0, 0, 255), 2)  # 红色框
#     img = cv2.resize(img, (400,400))
    save_path = os.path.join('./results', f"box_{img_path.split('/')[-1]}")
    cv2.imwrite(save_path, img)


def extract_boxes(input_string):
    # 正则表达式匹配 <box>(x1,y1),(x2,y2)</box>
    pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    
    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, input_string)
    
    # 将匹配到的字符串转换为四元组并存入列表
    box_list = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in matches]
    
    return box_list
'''
samples_root 测试图像路径
ckpt_dir 模型权重路径
'''
json_data = []
cnt=0
list = []
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
samples_root = './Y+H'
ckpt_dir = '/home/tione/notebook/lcs/codes/ms-swift-main/swift/llm/data/output/qwen-vl-chat/v34-20241216-001938/checkpoint-6000'
model_type = ModelType.qwen_vl_chat
template_type = get_default_template_type(model_type)

model_id_or_path = None
model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_id_or_path=model_id_or_path, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)

model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(20)

for folder in os.listdir(samples_root):
    item_list = os.listdir(os.path.join(samples_root, folder))
    item = item_list[0]
    print(item)
    if not item.endswith('.png') and  not item.endswith('.jpg'):
        continue
    cnt+=1
    if cnt == 200:
        break
    img_path = os.path.join(samples_root,folder,item)

#         query_0 = f"<img>{img_path}</img>We are attempting to decrypt an oracle bone picture. Experts have proposed several possible decryption directions, as follows:" + n2t[item] + "Count the number of distinct parts in this oracle bone script character and output this number."
#     print(query_0)
    query_0 = f"<img>{img_path}</img>Count the number of distinct parts in this oracle bone script character and output this number."
    response_0, history = inference(model, template, query_0)

    # 流式
    query_1 = f"Describe each distinct part of the oracle bone script character on the first line, If the character consists of multiple parts (e.g., left-right or top-bottom structure), provide a description for each part, separating them with commas. On the second line, provide a general description of the meaning of this oracle bone script image."
    print_idx=0
    gen = inference_stream(model, template, query_1, history)
    for response, history in gen:
        delta = response[print_idx:]
#             print(delta, end='', flush=True)
        print_idx = len(response)
    query_2 = f"Provide each distinct part of the oracle bone script character with the grounding on the first line."
    gen = inference_stream(model, template, query_2, history)
    for response, history in gen:
        delta = response[print_idx:]
#             print(delta, end='', flush=True)
        print_idx = len(response)
    print(f'history: {history}')
    json_entry = {
        "id": item,
        "conversations":[
            {"query": query_0, "predict": int(response_0)},
            {"query": query_1, "predict": history[1][1]},
            {"query": query_2, "predict": extract_boxes(history[2][1])}
        ]
    }
    json_data.append(json_entry)

    draw(img_path, extract_boxes(history[2][1]))

print(cnt)


# Save the json data to a file
json_output_path = './poyi/inference_poyi_result_6000.json'
with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON file saved as {json_output_path}")




