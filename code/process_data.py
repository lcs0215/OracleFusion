import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def image_to_beziers(image_path, threshold=128, num_points=40):
    # 读取并预处理图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(224, 224))
    # 高斯模糊
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # 自适应阈值
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    beziers = []
    control_points = []
    def fit_bezier(contour, num_points):
        if len(contour) < 4:
            return None
        tck, u = splprep(contour[:, 0, :].T, s=3, per=True)
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck, der=0)
        return np.vstack((x_new, y_new)).T

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # 仅处理外部轮廓
            bezier = fit_bezier(contour, num_points)
            if bezier is not None:
                beziers.append(bezier)
                control_points.append(bezier)
            
            # 查找当前轮廓的内部孔洞
            hole_index = hierarchy[0][i][2]
            while hole_index != -1:
                hole_contour = contours[hole_index]
                hole_bezier = fit_bezier(hole_contour, num_points)
                if hole_bezier is not None:
                    beziers.append(hole_bezier)
                    control_points.append(hole_bezier)
                hole_index = hierarchy[0][hole_index][0]
    
    return beziers, control_points, contours

def plot_control_points(image_path, control_points, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(224, 224))
    plt.imshow(image, cmap='gray')
    for bezier in beziers:
        plt.plot(bezier[:, 0], bezier[:, 1], 'g-')
    for points in control_points:
        plt.scatter(points[:, 0], points[:, 1], c='r', s=4)
    plt.title("Control Points on Image")
    plt.savefig('see.png')
def save_contours_as_image(image_path, contours, output_file='contours.png'):
    # 读取原始图像
    img = cv2.imread(image_path)
    img = cv2.resize(img,(224, 224))
    # 绘制轮廓
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)  # 绿色轮廓，线条宽度为2

    # 保存带有轮廓的图像
    cv2.imwrite(output_file, img)
# 示例用法
image_path = 'datasets/image_rId10.png'
beziers, control_points, contours = image_to_beziers(image_path)
print(len(beziers))
plot_control_points(image_path, control_points)
save_contours_as_image(image_path, contours)