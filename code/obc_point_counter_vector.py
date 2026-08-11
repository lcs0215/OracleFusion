import cv2
import numpy as np
from PIL import Image
import torch
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier
import svgwrite
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from torch import nn
from scipy.interpolate import CubicSpline
from copy import deepcopy

def scale_and_offset_image(image, scale, offset_x, offset_y, new_width, new_height):
    # Step 1: Scale the image
    h, w = image.shape
    scaled_w, scaled_h = int(w * scale), int(h * scale)
    scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    
    # Step 2: Handle offset by cropping or padding
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    
    # If offset is negative, we need to crop the image
    if offset_x < 0:
        scaled_image = scaled_image[:, -offset_x:]  # Crop left side
    else:
        # Add padding to the right
        scaled_image = np.pad(scaled_image, ((0, 0), (offset_x, 0)), mode='constant', constant_values=255)
    
    if offset_y < 0:
        scaled_image = scaled_image[-offset_y:, :]  # Crop top side
    else:
        # Add padding to the bottom
        scaled_image = np.pad(scaled_image, ((offset_y, 0), (0, 0)), mode='constant', constant_values=255)
        
    scaled_h, scaled_w = scaled_image.shape[:2]
    # # Step 3: Create the final new size image and position the scaled/offset image onto it
    
    if new_height < scaled_h:
        scaled_image = scaled_image[:new_height]
    else:
        scaled_image = np.pad(scaled_image, ((0, new_height - scaled_h), (0, 0)), mode='constant', constant_values=255)
    
    if new_width < scaled_w:
        scaled_image = scaled_image[:, :new_width]
    else:
        scaled_image = np.pad(scaled_image, ((0, 0), (0, new_width - scaled_w)), mode='constant', constant_values=255)
    
    return scaled_image

def fanse(img):
    img = 255 - img
    return img

def neighbours1(x, y, img):
    h, w = img.shape
    n = []
    
    for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            n.append(img[ny, nx])
        else:
            n.append(0)
    return n

def neighbours(x, y, img):
    r2 = neighbours1(x, y ,img)
    return r2


def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

def ZhangSuenPlus03(image):

    image = fanse(image) / 255
    indexXY = np.argwhere(image > 0)
    minxy = np.min(indexXY, axis=0)
    maxxy = np.max(indexXY, axis=0)
    roi = image[max(minxy[0] - 1, 0):min(maxxy[0] + 2, image.shape[0]), 
                max(minxy[1] - 1, 0):min(maxxy[1] + 2, image.shape[1])]
    
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        indexXY = np.argwhere(roi > 0)
        if indexXY.size == 0:
            break
        minxy = np.min(indexXY, axis=0)
        maxxy = np.max(indexXY, axis=0)
        roi = roi[max(minxy[0] - 1, 0):min(maxxy[0] + 2, roi.shape[0]), 
                  max(minxy[1] - 1, 0):min(maxxy[1] + 2, roi.shape[1])]

        changing1 = []
        for y in range(1, len(roi) - 1):
            for x in range(1, len(roi[0]) - 1):
                if roi[y][x] == 1:
                    n = neighbours(x, y, roi)
                    P2, P3, P4, P5, P6, P7, P8, P9 = n
                    if (P4 * P6 * P8 == 0 and P2 * P4 * P6 == 0 and 
                        transitions(n) == 1 and 2 <= sum(n) <= 6):
                        changing1.append((x, y))
        for x, y in changing1:
            roi[y][x] = 0
        
        changing2 = []
        for y in range(1, len(roi) - 1):
            for x in range(1, len(roi[0]) - 1):
                if roi[y][x] == 1:
                    n = neighbours(x, y, roi)
                    P2, P3, P4, P5, P6, P7, P8, P9 = n
                    if (P2 * P6 * P8 == 0 and P2 * P4 * P8 == 0 and 
                        transitions(n) == 1 and 2 <= sum(n) <= 6):
                        changing2.append((x, y))
        for x, y in changing2:
            roi[y][x] = 0
    
    flags = roi > 0
    roi[flags] = 255
    return fanse(image)


# 获取到图中某个节点上的值
def getPointValue(maps,point):
    height, width = maps.shape[:2]
    if 0 < point[0] < height and 0 < point[1] < width:
        return maps[point[0],point[1]]
    else:
        return None

# 设置图上某个点的位置的值
def setPointValue(maps,point,value):
    maps[point[0],point[1]] = value


def IterMaps():
    point_maps = {
        "l": np.array([1, 0]),
        "r": np.array([-1, 0]),
        "u": np.array([0, 1]),
        "d": np.array([0, -1]),
        "lu": np.array([1, 1]),
        "ld": np.array([1, -1]),
        "ru": np.array([-1, 1]),
        "rd": np.array([-1, -1]),
    }
    
    def getPointMap():
        return point_maps

    # 使用栈实现路径遍历
    def iterateMaps(maps, start_point):
        stack = [start_point]  # 用栈来模拟递归
        points = []  # 存储遍历过的点

        setPointValue(maps, start_point, 2)  # 将起始点标记为已遍历
        points.append(start_point)

        while stack:
            point = stack.pop()

            # 遍历所有方向
            for d, d_point in getPointMap().items():
                new_point = point + d_point

                if getPointValue(maps, new_point) == 1:
                    setPointValue(maps, new_point, 2)  # 标记为已遍历
                    points.append(new_point)
                    
                    # 将之前的点以及现在的点都压入栈，用于路径复现
                    stack.append(point)
                    stack.append(new_point)  # 将新点压入栈中
                    #### 将路径进行保存
                    break

        return points

    return iterateMaps



def split_points_by_neighboor(points):
    up = points[0]
    
    point_group = []
    points_buffer = [up]
    point_group.append(points_buffer)
    
    for i in range(1, len(points)):
        dist_up2now = np.linalg.norm(points[i] - up)
        if dist_up2now < np.sqrt(2) + 0.1:
            points_buffer.append(points[i])
        else:
            points_buffer = [points[i]]
            point_group.append(points_buffer)

        up = points[i]
    point_group = [point for point in point_group if len(point) > 1]
    return point_group

# 计算方向向量并获得法向量
def calculate_normals(points, width):
    normals = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        v = p2 - p1
        # 计算法向量
        n = np.array([-v[1], v[0]])  # 旋转90度
        n = (n / np.linalg.norm(n)) * (width / 2)
        normals.append(n)
    # 添加最后一个点的法向量
    normals.append(normals[-1])  # 末点使用前一个点的法向量
    return normals

def smooth_normals(normals, windows_size=3):
    smoothed_normals = []
    for i in range(len(normals)):
        window = normals[max(0, i - windows_size):min(len(normals), i + windows_size + 1)]
        smoothed_normal = np.mean(window, axis=0)
        smoothed_normals.append(smoothed_normal)
    return smoothed_normals

class ObcVectorUtils:
    def __init__(self, image_path, resize_width = 512,control_point_sample_number = 5, line_width = 12):
        
        self.src_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resize_scale = resize_width / self.src_image.shape[1]
        
        self.resize_scale = resize_scale
        
        self.src_image = cv2.resize(self.src_image, (resize_width, int(self.src_image.shape[0] * resize_scale)))
        
        self.resize_sk_points = None
        self.resize_contours = None
        self.resize_src_image = None
        self.offest_x = None
        self.offest_y = None
        self.scale = None
        
        # 二值化
        _, self.thresh_image = cv2.threshold(self.src_image, 70, 255, cv2.THRESH_BINARY)
        
        # 提取骨架
        self.sk_thres_image = ZhangSuenPlus03(self.thresh_image)
        
        bin_image = np.where(self.sk_thres_image > 120,np.zeros_like(self.sk_thres_image),np.ones_like(self.sk_thres_image))
        
        bin_image = torch.from_numpy(bin_image)[None,None,:,:].float()
        # 定义卷积操作
        # 定义卷积块，使用卷积计算入口点
        conv = nn.Conv2d(1,1,3,1,1,bias = False)
        conv.weight.requires_grad = False
        # 定义卷积块中的算子
        conv.weight.copy_(torch.tensor(
            [[1,1,1],
            [1,10,1],
            [1,1,1]]
        ))
        
        check_result  = conv(bin_image)
        
        # 获取需要进行遍历的关键点
        out_res = torch.where(check_result == 11.0,torch.zeros_like(check_result),torch.ones_like(check_result))
        out_res = np.uint8(out_res.cpu().numpy()[0,0])
        
        """整个算法的流程为如下:
        首先随意选中一点，然后通过该点找到下一个点的坐标，然后通过该点不断进行消除原始图像中的节点，然后在那个点上放置一个flag。
        定义flag:
        0 表示为无字的区域
        1 表示原始像素点
        2 表示被消去的点"""
        
        # 获取所有起点的位置
        
        bin_image = np.where(self.sk_thres_image > 120,np.zeros_like(self.sk_thres_image),np.ones_like(self.sk_thres_image))
        start_positions = np.argwhere(out_res == 1)
        
        maps = np.uint8(bin_image)
        # 获取图的高度和宽度    `q`q
        height = maps.shape[0]
        width  = maps.shape[1]

        itemMaps = IterMaps()
        all_points = []
        for point in start_positions:
            if getPointValue(maps,point) == 1:
                point_now = point
                points = itemMaps(maps,point_now)
                
                point_group = split_points_by_neighboor(points)
                
                for points in point_group:
                    points = np.concatenate([point[None, None, :] for point in points], axis = 0)[:,:,::-1]
                    all_points.append(points)

        # 检查图中是否还有没有被检测的图像
        no_check_positions = np.argwhere(maps == 1)
        
        for point in no_check_positions:
            if getPointValue(maps,point) == 1:
                point_now = point
                points = itemMaps(maps,point_now)
                point_group = split_points_by_neighboor(points)
                
                for points in point_group:
                    points = np.concatenate([point[None, None, :] for point in points], axis = 0)[:,:,::-1]
                    all_points.append(points)
        self.sk_points = deepcopy(all_points)
        
        outline_smooths = []
        for index in range(len(all_points)):
            # 示例的连续点
            points = all_points[index][:, 0, :]
            # 计算左边缘和右边缘
            normals = calculate_normals(points, line_width)
            normals = smooth_normals(normals, windows_size = 10)
            left_outline = points + normals
            right_outline = points - normals

            # 生成样条曲线
            t = np.arange(len(points))  # 参数 t
            cs_left = CubicSpline(t, left_outline, axis=0)
            cs_right = CubicSpline(t, right_outline, axis=0)

            # 在曲线插值上生成平滑点
            t_fine = np.linspace(0, len(points) - 1, len(points) // control_point_sample_number)
            left_smooth = cs_left(t_fine)
            right_smooth = cs_right(t_fine)

            # 合并平滑后的左右轮廓
            outline_smooth = np.vstack([left_smooth, right_smooth[::-1]])
            if outline_smooth.shape[0] > 0:
                outline_smooths.append(outline_smooth[:, None, :])

        self.contours = outline_smooths
    
    
    def draw_split_contours(self):
        images = []
        
        for point in self.contours:
            new_maps = np.zeros_like(self.thresh_image)    
            for itempoint in point:
                setPointValue(new_maps,itempoint[0][::-1],1)
            
            images.append(Image.fromarray(new_maps * 255))
            
        new_image = Image.new(mode="L",size=(images[0].width * len(self.contours),images[0].width))
        
        for index,image in enumerate(images):
            new_image.paste(image,box=(image.width * index,0))
        return new_image
    
    def reduce_sk_points(self, sk_points, n):
        return [sk_point_set[::n] for sk_point_set in sk_points]
    
    def compute_contour_color(self):
        contour_color = []
        for i in range(len(self.contours)):
            
            # 创建一个与图像相同大小的空白掩码
            mask = np.zeros(self.src_image.shape[:2], dtype=np.uint8)
            
            # 使用轮廓填充掩码
            cv2.drawContours(mask, [self.contours[i]], -1, 255, thickness=cv2.FILLED)
            
            for j in range(i + 1, len(self.contours)):
                cv2.drawContours(mask, [self.contours[j]], -1, 0, thickness=cv2.FILLED)

            mean_val = np.mean(self.thresh_image[mask == 255])
            
            if mean_val > 200:
                contour_color.append('black')
            else:
                contour_color.append('white')
        return contour_color
    
    def get_new_countour_by_size(self, new_size, graph_size):
        contours = self.contours

        # 将轮廓缩放到指定大小
        max_x, max_y = 0, 0
        min_x, min_y = float('inf'), float('inf')

        # 找到轮廓的最大和最小坐标
        for contour in contours:
            max_x = max(max_x, np.max(contour[:, 0, 0]))
            max_y = max(max_y, np.max(contour[:, 0, 1]))
            min_x = min(min_x, np.min(contour[:, 0, 0]))
            min_y = min(min_y, np.min(contour[:, 0, 1]))

        # 计算原始轮廓的宽度和高度
        original_width = max_x - min_x
        original_height = max_y - min_y

        # 计算缩放因子
        scale_x = graph_size[0] / original_width
        scale_y = graph_size[1] / original_height
        scale = min(scale_x, scale_y)  # 保持纵横比不变

        # 计算新的轮廓大小
        new_width = original_width * scale
        new_height = original_height * scale

        # 计算图形的居中偏移
        offset_x = (new_size[0] - new_width) / 2
        offset_y = (new_size[1] - new_height) / 2

        new_contours = []
        for contour in contours:
            # 缩放并平移轮廓
            scaled_contour = contour.copy()
            scaled_contour[:, 0, 0] = (scaled_contour[:, 0, 0] - min_x) * scale + offset_x
            scaled_contour[:, 0, 1] = (scaled_contour[:, 0, 1] - min_y) * scale + offset_y
            new_contours.append(scaled_contour)
        
        new_sk_points = []
        for contour in self.sk_points:
            scaled_contour = deepcopy(contour)
            scaled_contour[:, 0, 0] = (scaled_contour[:, 0, 0] - min_x) * scale + offset_x
            scaled_contour[:, 0, 1] = (scaled_contour[:, 0, 1] - min_y) * scale + offset_y
            new_sk_points.append(scaled_contour)
            
            
        self.resize_sk_points = new_sk_points
        self.resize_contours = new_contours
        
        # 获取resize之后的图片
        image = self.src_image
        offset_x = -min_x * scale + offset_x
        offset_y = -min_y * scale + offset_y
        
        self.resize_src_image = scale_and_offset_image(image, 
                                                scale = scale, 
                                                offset_x = int(offset_x),
                                                offset_y = int(offset_y), 
                                                new_width = new_size[0],
                                                new_height = new_size[1])
        self.offest_x = offset_x
        self.offest_y = offset_y
        self.scale = self.resize_scale * scale
        
        return new_contours

    def save_path_to_svg(self, svg_paths, filename='bezier_curve.svg', size = (600, 600)):
        # 创建一个 svg 文件
        dwg = svgwrite.Drawing(filename, profile='tiny', size = size)
        
        for svg_path in svg_paths:
            # 作为贝塞尔曲线的起点以及终点
            bezier_start_point = svg_path[0].start
            
            # 定义路径字符串
            path_data = [f"M {bezier_start_point.real},{bezier_start_point.imag} "]
            
            for segment in svg_path:
                # 控制点和终点
                control1 = segment.control1
                control2 = segment.control2
                end = segment.end
                
                # 添加到路径数据中，生成 SVG path 的 d 属性格式
                path_data.append(f'C {control1.real},{control1.imag}, {control2.real},{control2.imag}, {end.real},{end.imag}')
            
            # 将路径数据合并成字符串
            path_string = " ".join(path_data)
            
            # 创建一个 SVG 路径对象并添加到文件
            dwg.add(dwg.path(d=path_string, stroke = "none", fill = "black", stroke_width=1))

        # 保存文件
        dwg.save(True)
        print(f"SVG file saved as {filename}")
    
    
    def to_svg(self, output_path, size = (600, 600), center_size = (512, 512)):
        contours = self.get_new_countour_by_size(size, center_size)
        
        paths = []
        for points in contours:
            points = points[:, 0, :]
            
            
            path = Path()
            # 将顶点连成直线，然后替换为贝塞尔曲线
            # 作为贝塞尔曲线的起点以及终点
            bezier_start_point = points[0]

            for i in range(len(points) - 1):
                start = complex(points[i][0], points[i][1])
                end = complex(points[i+1][0], points[i+1][1])
                
                # 用控制点来定义贝塞尔曲线，下面是手动设置
                dx = end.real - start.real
                dy = end.imag - start.imag
                
                control_point1 = complex(start.real + dx / 3, start.imag + dy / 3)
                control_point2 = complex(start.real + dx / 3 * 2, start.imag + dy / 3 * 2)
                
                # control_point1 = complex((start.real + end.real) / 2, start.imag)
                # control_point2 = complex(end.real, (start.imag + end.imag) / 2)

                
                # 添加三次贝塞尔曲线
                path.append(CubicBezier(start, control_point1, control_point2, end))

            start = end
            end = complex(bezier_start_point[0], bezier_start_point[1])
            control_point1 = complex((start.real + end.real) / 2, start.imag)
            control_point2 = complex(end.real, (start.imag + end.imag) / 2)

            # 添加最后一步的贝塞尔曲线
            path.append(CubicBezier(start, control_point1, control_point2, end))
            
            paths.append(path)
            
        self.save_path_to_svg(paths, output_path, size = size)
        
if __name__ == "__main__":
    src_image = cv2.imread("figures/舟.png")
    
    obc_utils = ObcVectorUtils("figures/舟.png",
                               control_point_sample_number = 30,
                               line_width = 18)
    
    ### NOTE : to_svg的时候对象会自动调用get_new_countour_by_size
    ### 这个是否会对图片进行放缩
    ### 所以这个是否box的位置也因该发生对应的变化
    obc_utils.to_svg("test.svg", center_size = (480, 480))
    
    # 应该先 对 box 进行 scale
    print(obc_utils.scale)
    
    # 然后分别对 x 坐标和 y 坐标进行 offest
    print(obc_utils.offest_x)
    print(obc_utils.offest_y)
    
    ### e.g
    box = np.array( [233.0, 77.0, 879.0, 999.0])

    cv2.imwrite("test.png", src_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
    box = box * obc_utils.scale
    box = box + np.array([obc_utils.offest_x, obc_utils.offest_y, obc_utils.offest_x, obc_utils.offest_y])
    # 限制坐标不能比0小
    box = np.clip(box, 0, None)
    print(box[2:] - box[:2])
    print(box)
    cv2.imwrite("test2.png", obc_utils.resize_src_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
    cv2.imwrite("test2_src.png", obc_utils.resize_src_image)
    
    