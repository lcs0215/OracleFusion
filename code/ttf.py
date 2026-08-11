from importlib import reload
import os
import numpy as np
import bezier
import freetype as ft
import pydiffvg
import torch
import save_svg
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from obc_point_counter_vector import ObcVectorUtils
device = torch.device("cuda" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

reload(bezier)
def plot_beziers(beziers, output_file="bezier_curves.png"):
    """ Plot and save the bezier curves. """
    plt.figure(figsize=(8, 8))
    
    for bez in beziers[0]:
        bez_points = np.array(bez)
        plt.plot(bez_points[:, 0], bez_points[:, 1], 'b-')
    
    plt.title('Bezier Curves')
    plt.gca().invert_yaxis()  # Invert y-axis to match typical font rendering
    plt.savefig(output_file)
    plt.show()

def fix_single_svg(svg_path, all_word=False):
    target_h_letter = 360
    target_canvas_width, target_canvas_height = 600, 600

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)

    letter_h = canvas_height
    letter_w = canvas_width

    if all_word:
        if letter_w > letter_h:
            scale_canvas_w = target_h_letter / letter_w
            hsize = int(letter_h * scale_canvas_w)
            scale_canvas_h = hsize / letter_h
        else:
            scale_canvas_h = target_h_letter / letter_h
            wsize = int(letter_w * scale_canvas_h)
            scale_canvas_w = wsize / letter_w
    else:
        scale_canvas_h = target_h_letter / letter_h
        wsize = int(letter_w * scale_canvas_h)
        scale_canvas_w = wsize / letter_w

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h + target_h_letter

    w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max([torch.max(p.points[:, 0]) for p in shapes])
    h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max([torch.max(p.points[:, 1]) for p in shapes])

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] + target_canvas_width/2 - int(w_min + (w_max - w_min) / 2)
        p.points[:, 1] = p.points[:, 1] + target_canvas_height/2 - int(h_min + (h_max - h_min) / 2)

    output_path = f"{svg_path[:-4]}_scaled.svg"
    save_svg.save_svg(output_path, target_canvas_width, target_canvas_height, shapes, shape_groups)


def normalize_letter_size(image_dir, dest_path, font_path, word):
    fontname = os.path.splitext(os.path.basename(font_path))[0]
    # for i, c in enumerate(txt):
        # fname = fname.replace(" ", "_")
    # fix_single_svg(fname)
    
    fname = f"{dest_path}/{fontname}_{word}.svg"
    fname = fname.replace(" ", "_")
    fix_single_svg(fname, all_word=True)


def bezier_chain_to_commands(C, closed=True):
    curves = bezier.chain_to_beziers(C)
    cmds = 'M %f %f ' % (C[0][0], C[0][1])
    n = len(curves)
    for i, bez in enumerate(curves):
        if i == n - 1 and closed:
            cmds += 'C %f %f %f %f %f %fz ' % (*bez[1], *bez[2], *bez[3])
        else:
            cmds += 'C %f %f %f %f %f %f ' % (*bez[1], *bez[2], *bez[3])
    return cmds


    

def count_cp(file_name, font_name):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_name)
    p_counter = 0
    for path in shapes:
        p_counter += path.points.shape[0]
    print(f"TOTAL CP:   [{p_counter}]")
    return p_counter


def write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path):
    cmds = ''
    svg = header

    path = '<g><path d="'
    for C in beziers:
        if subdivision_thresh is not None:
            print('subd')
            C = bezier.subdivide_bezier_chain(C, subdivision_thresh)
        cmds += bezier_chain_to_commands(C, True)
    path += cmds + '"/>\n'
    svg += path + '</g></svg>\n'

    fname = f"{dest_path}/{fontname}_{c}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg)
    f.close()
    return fname, path



def bezier_chain_to_commands(C, closed=True):
    curves = bezier.chain_to_beziers(C)
    cmds = 'M %f %f ' % (C[0][0], C[0][1])
    n = len(curves)
    for i, bez in enumerate(curves):
        if i == n - 1 and closed:
            cmds += 'C %f %f %f %f %f %fz ' % (*bez[1], *bez[2], *bez[3])
        else:
            cmds += 'C %f %f %f %f %f %f ' % (*bez[1], *bez[2], *bez[3])
    return cmds
def write_image_svg(image_path, beziers, dest_path):
    cmds = ''
    svg = '''<?xml version="1.0" encoding="utf-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" baseProfile="full" 
    width="600" height="600" viewBox="0 0 600 600">
    <defs/>
    <g>
    '''

    path = '<path d="'
    for C in beziers:
        cmds += bezier_chain_to_commands(C, True)
    path += cmds + '"/>\n'
    svg += path + '</g></svg>\n'

    fname = os.path.join(dest_path, os.path.basename(image_path).replace('.png', '.svg'))
    with open(fname, 'w') as f:
        f.write(svg)
    return fname



def image_to_svgs(image_dir, dest_path, font_path, word, bbox=None):

    ''' Load a font and convert the outlines for a given string to cubic bezier curves,
        if merge is True, simply return a list of all bezier curves,
        otherwise return a list of lists with the bezier curves for each glyph'''
    image_path  = os.path.join(image_dir, word) + ".png"
    if not os.path.exists(image_path):
        image_path  = os.path.join(image_dir, word) + ".jpg"
    src_image = cv2.imread(image_path)
    obc_utils = ObcVectorUtils(image_path,
                               control_point_sample_number = 6,
                               line_width = 16)
    
    ### NOTE : to_svg的时候对象会自动调用get_new_countour_by_size
    ### 这个是否会对图片进行放缩
    ### 所以这个是否box的位置也因该发生对应的变化
    obc_utils.to_svg(os.path.join(dest_path, f"{word}_scaled.svg"))

    
    # 应该先 对 bbox 进行 scale
    # print(obc_utils.scale)
    
    # 然后分别对 x 坐标和 y 坐标进行 offest
    # print(obc_utils.offest_x)
    # print(obc_utils.offest_y)

    if bbox != None:
        # cv2.imwrite("test.png", src_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        bbox = np.array(bbox)
        bbox = bbox * obc_utils.scale
        bbox = bbox + np.array([obc_utils.offest_x, obc_utils.offest_y, obc_utils.offest_x, obc_utils.offest_y])
        bbox = np.clip(bbox, 0, 600)
        # print(bbox)
        # cv2.imwrite("test2.png", obc_utils.resize_src_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
    return bbox, obc_utils



if __name__ == '__main__':

    fonts = ["KaushanScript-Regular"]
    level_of_cc = 1

    if level_of_cc == 0:
        target_cp = None

    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }

        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    for f in fonts:
        print(f"======= {f} =======")
        font_path = f"data/fonts/{f}.ttf"
        output_path = f"data/init"
        txt = "BUNNY"
        subdivision_thresh = None
        image_to_svgs(output_path, font_path, txt, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(output_path, font_path, txt)

        print("DONE")




