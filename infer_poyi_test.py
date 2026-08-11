import os
import pandas as pd
import json
import uuid
from PIL import Image
import os
import re
import numpy as np
import random
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
    color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0,255,255) , (128,0,128), (128,0,128), (128,0,128), (128,0,128)]
    for i,bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        print(x_min, y_min, x_max, y_max)
        cv2.rectangle(img, (int(w * x_min/1000), int(h * y_min/1000)), (int(w * x_max/1000), int(h*y_max/1000)), color_list[i], 2)  # 红色框
#     img = cv2.resize(img, (400,400))
    save_path = os.path.join('./poyi_visul', f"box_{img_path.split('/')[-1]}")
    cv2.imwrite(save_path, img)
# Define the root directory of the samples
samples_root_train = '/home/tione/notebook/lcs/codes/LLaMA-Factory/data/output/train_bbox'
samples_root_test = '/home/tione/notebook/lcs/codes/LLaMA-Factory/data/output/test_bbox'
json_file_path = './datasets/dataset'
box_dict={}
json_data = []
cnt=0
list = []
val_list =['鬽','赤','灾','狐','乡','天','沈','羞','豚','遝','尨','糸','涛','光','兵','咎','奴','竝','耋','重','阩','𠬝']
json_list =[item for item in os.listdir(json_file_path) if item.endswith('.json')]
with open('data_info.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('./datasets/desc_box.json', 'r', encoding='utf-8') as f:
    box_info = json.load(f)
for item in box_info:
    box_dict[item['code']] = item['points']

def extract_boxes(input_string):
    # 正则表达式匹配 <box>(x1,y1),(x2,y2)</box>
    pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    
    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, input_string)
    
    # 将匹配到的字符串转换为四元组并存入列表
    box_list = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in matches]
    
    return box_list

json_data = []
cnt=0
list = []
ckpt_dir = 'output/qwen-vl-chat/26-20241017-115933/checkpoint-6500'
model_type = ModelType.qwen_vl_chat
template_type = get_default_template_type(model_type)

model_id_or_path = None
model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_id_or_path=model_id_or_path, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)

model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(80)
# print(concept)
# random.seed(0)
for item in data:
    word = item['word']
    if word not in val_list:
        continue
    concept = item['concept']
    concept_list = item['texts']
    CONCEPT = ''
    image_path = ''
    if os.path.exists(os.path.join(samples_root_train, word) + '.png'):
        image_path = os.path.join(samples_root_train, word) + '.png'
    elif os.path.exists(os.path.join(samples_root_test, word) + '.png'):  
        image_path = os.path.join(samples_root_test, word) + '.png'
    else:
        print(f"{word}不存在")
        continue
#     item_list = [item for item in os.listdir(image_dir) if item.endswith('.png') and not item.endswith('aug.png')]
#     print(item_list)
#     item_list = random.sample(item_list, min(5, len(item_list)))
#     for file in item_list:
    img_pillow = Image.open(image_path)
    img_width = img_pillow.width  # 图片宽度
    img_height = img_pillow.height  # 图片高度
    box_list = []
    bbox = np.array(box_dict[word])
    bbox = (bbox * 1000 / img_width).astype(int)
    responce = ""
    for k, text in enumerate(concept_list):
        text=text.replace("'", '')
        text=text.replace('"', '')
        text=text.replace(",", '')
        text=text.replace("+", ' and ')
        text=text.replace("(", '')
        text=text.replace(")", '')
        text=text.replace(":", '')
        CONCEPT = CONCEPT + text
        if k != len(concept_list)-1:
            CONCEPT = CONCEPT + ', '

        box =[bbox[k][0][0].item(), bbox[k][0][1].item(), bbox[k][2][0].item(), bbox[k][2][1].item()]

        box_list.append(box)
#         print(box)
        img_path = image_path

        responce += f"<ref>{text}</ref><box>({box[0]},{box[1]}),({box[2]},{box[3]})</box> "
        if k != len(concept_list)-1:
            responce += ", "
        # print(responce)
        # Create a JSON object for each image
    cnt+=1
    CONCEPT = CONCEPT + "\n" + concept
#     print(box_list)

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
        "id": word,
        "conversations":[
            {"query": query_0, "predict": int(response_0), "label": len(concept_list)},
            {"query": query_1, "predict": history[1][1], "label": CONCEPT},
            {"query": query_2, "predict": extract_boxes(history[2][1]), "label": box_list}
        ]
    }
    json_data.append(json_entry)
    draw(img_path, extract_boxes(history[2][1]))

print(cnt)


# Save the json data to a file
json_output_path = './poyi/inference_result_poyi.json'
with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON file saved as {json_output_path}")




