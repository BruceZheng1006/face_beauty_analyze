import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import glob
import sys
import io
import cv2
import numpy as np

# 作为脚本
# face_landmarker.task下载链接
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 强制标准输出/错误输出使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def get_face_density(landmarks, img_w, img_h):
    # 将归一化坐标转换为像素坐标
    def to_pixel(idx):
        return [landmarks[idx].x * img_w, landmarks[idx].y * img_h]

    # --- A. 定义内圈 (五官密集区) ---
    # 选点：眉心(8), 左眼外角(33), 左嘴角(61), 下唇底(18), 右嘴角(291), 右眼外角(263)
    inner_indices = [8, 33, 61, 18, 291, 263]
    inner_pts = np.array([to_pixel(i) for i in inner_indices], dtype=np.int32)

    # --- B. 定义外圈 (全脸轮廓) ---
    # 选取面部边缘的代表点 (或使用 MediaPipe 提供的 FACE_CONTOURS 列表)
    outer_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    outer_pts = np.array([to_pixel(i) for i in outer_indices], dtype=np.int32)

    # 计算面积 (OpenCV 现成函数)
    area_inner = cv2.contourArea(inner_pts)
    area_outer = cv2.contourArea(outer_pts)

    # 计算比例 (密度值)
    density_ratio = area_inner / area_outer

    return density_ratio

# --- 几何计算工具函数 ---
def get_distance(p1, p2, img_w, img_h):
    return math.sqrt(((p1.x - p2.x) * img_w) ** 2 + ((p1.y - p2.y) * img_h) ** 2)


# 三点计算角度
def get_angle(p1, p2, p3, img_w, img_h):
    a = [p1.x * img_w, p1.y * img_h]
    b = [p2.x * img_w, p2.y * img_h]
    c = [p3.x * img_w, p3.y * img_h]
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)


# 两点计算和x轴成的角 第一个点是顶点
def get_twopoint_angle(p1, p2, img_w, img_h):
    # 转换关键点为图像像素坐标
    a = [p1.x * img_w, p1.y * img_h]
    b = [p2.x * img_w, p2.y * img_h]

    # 计算弧度并转换为角度，取绝对值
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)


# --- 封装单张图片处理函数 ---
def process_single_image(image_path, detector, face_key_points, number):
    """
    处理单张人脸图片，输出美学测量结果
    :param image_path: 单张图片的完整路径
    :param detector: 初始化好的人脸检测器
    :param face_key_points: 人脸关键点映射字典
    """
    try:
        # 打印当前处理的图片名称
        image_name = os.path.basename(image_path)
        print(f"\n==================================================")
        print(f"正在处理图片：{image_name}")
        print(f"==================================================")

        # 加载图片
        image = mp.Image.create_from_file(image_path)
        img_h, img_w = image.height, image.width
        result = detector.detect(image)

        if result.face_landmarks:
            for face_idx, landmarks in enumerate(result.face_landmarks):
                print(f"\n========== 第 {number} 张人脸 (基础坐标) ==========")
                for point_name, point_index in face_key_points.items():
                    if point_index < len(landmarks):
                        point = landmarks[point_index]
                        print(f"🔹 {point_name}（索引{point_index}）：X={point.x:.4f}, Y={point.y:.4f}, Z={point.z:.4f}")

                # --- 比例尺计算 ---
                px_ipd = get_distance(landmarks[468], landmarks[473], img_w, img_h)
                factor = 63.0 / px_ipd
                print(f"\n========== 深度美学测量 (单位: mm) ==========")

                # 1. 三庭
                print(f"\n● 三庭")
                top_h = get_distance(landmarks[10], landmarks[9], img_w, img_h) * factor
                mid_h = get_distance(landmarks[9], landmarks[2], img_w, img_h) * factor
                bot_h = get_distance(landmarks[2], landmarks[152], img_w, img_h) * factor
                total_h = top_h + mid_h + bot_h

                print(f"三庭比例: {top_h / total_h:.2f}:{mid_h / total_h:.2f}:{bot_h / total_h:.2f}")
                print(f"上庭长度: {top_h:.2f} mm")
                print(f"上庭占比: {top_h / total_h:.2f}")
                print(f"中庭长度: {mid_h:.2f} mm")
                mid_ratio = mid_h / total_h
                print(f"中庭占比: {mid_ratio:.2f}")
                print(f"下庭长度: {bot_h:.2f} mm")
                print(f"下庭占比: {bot_h / total_h:.2f}")
                down_ratio = bot_h / total_h
                mid_thresholds = [0.38, 0.42, 0.44, 0.46, 0.47, 0.5]
                if mid_ratio < mid_thresholds[0]:
                    mid_grade = "中庭超小"
                elif mid_thresholds[0] <= mid_ratio < mid_thresholds[1]:
                    mid_grade = "中庭小"
                elif mid_thresholds[1] <= mid_ratio < mid_thresholds[2]:
                    mid_grade = "中庭稍小"
                elif mid_thresholds[2] <= mid_ratio < mid_thresholds[3]:
                    mid_grade = "中庭标准"
                elif mid_thresholds[3] <= mid_ratio < mid_thresholds[4]:
                    mid_grade = "中庭稍大"
                elif mid_thresholds[4] <= mid_ratio < mid_thresholds[5]:
                    mid_grade = "中庭大"
                else:
                    mid_grade = "中庭超大"
                print(f"中庭尺寸评价: {mid_grade}")

                down_thresholds = [0.28, 0.32, 0.34, 0.37, 0.39, 0.43]
                if down_ratio < down_thresholds[0]:
                    down_grade = "下庭超小"
                elif down_thresholds[0] <= down_ratio < down_thresholds[1]:
                    down_grade = "下庭小"
                elif down_thresholds[1] <= down_ratio < down_thresholds[2]:
                    down_grade = "下庭稍小"
                elif down_thresholds[2] <= down_ratio < down_thresholds[3]:
                    down_grade = "下庭标准"
                elif down_thresholds[3] <= down_ratio < down_thresholds[4]:
                    down_grade = "下庭稍大"
                elif down_thresholds[4] <= down_ratio < down_thresholds[5]:
                    down_grade = "下庭大"
                else:
                    down_grade = "下庭超大"
                print(f"下庭尺寸评价: {down_grade}")
                # 2. 五眼
                print(f"\n● 五眼")
                # 顺序：图片左侧(右眼外侧) -> 图片右侧(左眼外侧)
                dist_r_out = get_distance(landmarks[234], landmarks[33], img_w, img_h) * factor
                dist_r_eye = get_distance(landmarks[33], landmarks[133], img_w, img_h) * factor
                dist_mid = get_distance(landmarks[133], landmarks[362], img_w, img_h) * factor
                dist_l_eye = get_distance(landmarks[362], landmarks[263], img_w, img_h) * factor
                dist_l_out = get_distance(landmarks[263], landmarks[454], img_w, img_h) * factor
                print(f"五眼比例: {dist_r_out / dist_mid:.2f}:{dist_r_eye / dist_mid:.2f}:1.00:{dist_l_eye / dist_mid:.2f}:{dist_l_out / dist_mid:.2f}")
                print(f"右眼外侧留白距离: {dist_r_out:.2f} mm")
                print(f"右眼外侧留白距离占比: {dist_r_out / dist_mid:.2f}")
                print(f"右眼长度: {dist_r_eye:.2f} mm")
                print(f"内眼角间距: {dist_mid:.2f} mm")  # 通常作为基准1
                print(f"内眼角间距占比: {dist_mid / dist_mid:.2f}")
                print(f"左眼长度: {dist_l_eye:.2f} mm")
                print(f"左眼外侧留白距离: {dist_l_out:.2f} mm")
                print(f"左眼外侧留白距离占比: {dist_l_out / dist_mid:.2f}")

                eye_width_thresholds = [22.579, 24.688, 25.9555, 28.9536, 30.8444, 33.742]
                # 提取右眼宽度原始值
                r_eye_width = dist_r_eye
                # 7个区间的判断逻辑
                if r_eye_width < eye_width_thresholds[0]:
                    eye_grade = "眼睛超短"
                elif eye_width_thresholds[0] <= r_eye_width < eye_width_thresholds[1]:
                    eye_grade = "眼睛短"
                elif eye_width_thresholds[1] <= r_eye_width < eye_width_thresholds[2]:
                    eye_grade = "眼睛稍短"
                elif eye_width_thresholds[2] <= r_eye_width < eye_width_thresholds[3]:
                    eye_grade = "眼睛标准"
                elif eye_width_thresholds[3] <= r_eye_width < eye_width_thresholds[4]:
                    eye_grade = "眼睛稍长"
                elif eye_width_thresholds[4] <= r_eye_width < eye_width_thresholds[5]:
                    eye_grade = "眼睛长"
                else:
                    eye_grade = "眼睛超长"

                # 你的6个内眼角间距阈值（已按从小到大排序）
                eye_dist_thresholds = [32.583, 34.04, 34.9855, 36.4, 37.0896, 38.65]

                # 提取内眼角间距原始值
                inner_eye_dist = dist_mid

                # 7个区间的判断逻辑
                if inner_eye_dist < eye_dist_thresholds[0]:
                    eye_dist_grade = "眼距超小"
                elif eye_dist_thresholds[0] <= inner_eye_dist < eye_dist_thresholds[1]:
                    eye_dist_grade = "眼距小"
                elif eye_dist_thresholds[1] <= inner_eye_dist < eye_dist_thresholds[2]:
                    eye_dist_grade = "眼距稍小"
                elif eye_dist_thresholds[2] <= inner_eye_dist < eye_dist_thresholds[3]:
                    eye_dist_grade = "眼距标准"
                elif eye_dist_thresholds[3] <= inner_eye_dist < eye_dist_thresholds[4]:
                    eye_dist_grade = "眼距稍大"
                elif eye_dist_thresholds[4] <= inner_eye_dist < eye_dist_thresholds[5]:
                    eye_dist_grade = "眼距大"
                else:
                    eye_dist_grade = "眼距超大"

                # 输出内眼角间距评价结果
                print(f"内眼角间距评价: {eye_dist_grade}")


                # 3. 脸部
                print(f"\n● 脸部")
                face_length = get_distance(landmarks[10], landmarks[152], img_w, img_h) * factor
                forehead_width = get_distance(landmarks[127], landmarks[356], img_w, img_h) * factor  # 颞部宽度
                zygomatic_width = get_distance(landmarks[234], landmarks[454], img_w, img_h) * factor  # 颧骨宽度
                mandible_width = get_distance(landmarks[172], landmarks[397], img_w, img_h) * factor  # 下颌角宽度
                mandible_angle = get_angle(landmarks[361], landmarks[397], landmarks[152], img_w, img_h)  # 下颌角度数
                upface_height = get_distance(landmarks[9], landmarks[0], img_w, img_h) * factor  # 上脸高度
                print(f"脸部长度: {face_length:.2f} mm")
                print(f"颞部宽度: {forehead_width:.2f} mm")
                print(f"颧骨宽度: {zygomatic_width:.2f} mm")
                print(f"下颌角宽度: {mandible_width:.2f} mm")
                print(f"下颌角度数: {mandible_angle:.2f} °")
                print(
                    f"颞宽:颧宽:下颌宽: {forehead_width / zygomatic_width:.2f}:1.00:{mandible_width / zygomatic_width:.2f}")
                # fWHR:颧骨宽度 / 上脸高度(眉毛中心点到上唇上缘)
                fwhr = zygomatic_width / upface_height
                print(f"fWHR(面部宽高比): {fwhr:.2f}")
                # 你的6个脸部长度阈值（已按从小到大排序）
                face_len_thresholds = [140.322, 150.8276, 156.56, 167.0112, 173.6512, 193.292]

                # 提取脸部长度原始值
                current_face_len = face_length

                # 7个区间的判断逻辑
                if current_face_len < face_len_thresholds[0]:
                    face_len_grade = "脸超短"
                elif face_len_thresholds[0] <= current_face_len < face_len_thresholds[1]:
                    face_len_grade = "脸短"
                elif face_len_thresholds[1] <= current_face_len < face_len_thresholds[2]:
                    face_len_grade = "脸稍短"
                elif face_len_thresholds[2] <= current_face_len < face_len_thresholds[3]:
                    face_len_grade = "脸长标准"
                elif face_len_thresholds[3] <= current_face_len < face_len_thresholds[4]:
                    face_len_grade = "脸稍长"
                elif face_len_thresholds[4] <= current_face_len < face_len_thresholds[5]:
                    face_len_grade = "脸长"
                else:
                    face_len_grade = "脸超长"

                # 输出脸部长度评价结果
                print(f"脸部长度评价: {face_len_grade}")


                zygomatic_thresholds = [119.428,124.83, 127.96, 135.38, 139.3292, 149.678]
                current_zygomatic = zygomatic_width
                if current_zygomatic < zygomatic_thresholds[0]:
                    zygomatic_grade = "颧骨超短"
                elif zygomatic_thresholds[0] <= current_zygomatic < zygomatic_thresholds[1]:
                    zygomatic_grade = "颧骨短"
                elif zygomatic_thresholds[1] <= current_zygomatic < zygomatic_thresholds[2]:
                    zygomatic_grade = "颧骨稍短"
                elif zygomatic_thresholds[2] <= current_zygomatic < zygomatic_thresholds[3]:
                    zygomatic_grade = "颧骨标准"
                elif zygomatic_thresholds[3] <= current_zygomatic < zygomatic_thresholds[4]:
                    zygomatic_grade = "颧骨稍长"
                elif zygomatic_thresholds[4] <= current_zygomatic < zygomatic_thresholds[5]:
                    zygomatic_grade = "颧骨长"
                else:
                    zygomatic_grade = "颧骨超长"
                print(f"颧骨宽度评价: {zygomatic_grade}")

                # --- 下颌角度数区间判断（表格数值：137.287、141.0352、143.6255、148.69、152.6624、160.81）---
                mandible_angle_thresholds = [137.287, 141.0352, 143.6255, 148.69, 152.6624, 160.81]
                if mandible_angle < mandible_angle_thresholds[0]:
                    mandible_angle_grade = "下颌角超小"
                elif mandible_angle_thresholds[0] <= mandible_angle < mandible_angle_thresholds[1]:
                    mandible_angle_grade = "下颌角小"
                elif mandible_angle_thresholds[1] <= mandible_angle < mandible_angle_thresholds[2]:
                    mandible_angle_grade = "下颌角稍小"
                elif mandible_angle_thresholds[2] <= mandible_angle < mandible_angle_thresholds[3]:
                    mandible_angle_grade = "下颌角标准"
                elif mandible_angle_thresholds[3] <= mandible_angle < mandible_angle_thresholds[4]:
                    mandible_angle_grade = "下颌角稍大"
                elif mandible_angle_thresholds[4] <= mandible_angle < mandible_angle_thresholds[5]:
                    mandible_angle_grade = "下颌角大"
                else:
                    mandible_angle_grade = "下颌角超大"
                print(f"下颌角度数评价: {mandible_angle_grade}")

                # --- fWHR区间判断（表格数值：1.33、1.43、1.48、1.59、1.66、1.81）---
                fwhr_thresholds = [1.33, 1.43, 1.48, 1.59, 1.66, 1.81]
                if fwhr < fwhr_thresholds[0]:
                    fwhr_grade = "面宽超小"
                elif fwhr_thresholds[0] <= fwhr < fwhr_thresholds[1]:
                    fwhr_grade = "面宽小"
                elif fwhr_thresholds[1] <= fwhr < fwhr_thresholds[2]:
                    fwhr_grade = "面宽稍小"
                elif fwhr_thresholds[2] <= fwhr < fwhr_thresholds[3]:
                    fwhr_grade = "面宽标准"
                elif fwhr_thresholds[3] <= fwhr < fwhr_thresholds[4]:
                    fwhr_grade = "面宽稍大"
                elif fwhr_thresholds[4] <= fwhr < fwhr_thresholds[5]:
                    fwhr_grade = "面宽大"
                else:
                    fwhr_grade = "面宽超大"
                print(f"面部宽高比评价: {fwhr_grade}")

                # 4. 黄金三角
                print(f"\n● 黄金三角")
                # 顶点为鼻尖(1)，两底点为左右瞳孔(473,468)
                golden_angle = get_angle(landmarks[468], landmarks[1], landmarks[473], img_w, img_h)
                print(f"黄金三角度数: {golden_angle:.2f} °")
                # --- 黄金三角度数区间判断（表格数值：54.162、60.42、63.59、68.5224、71.8872、81.452）---
                golden_angle_thresholds = [54.162, 60.42, 63.59, 68.5224, 71.8872, 81.452]
                if golden_angle < golden_angle_thresholds[0]:
                    golden_angle_grade = "黄金三角超小"
                elif golden_angle_thresholds[0] <= golden_angle < golden_angle_thresholds[1]:
                    golden_angle_grade = "黄金三角小"
                elif golden_angle_thresholds[1] <= golden_angle < golden_angle_thresholds[2]:
                    golden_angle_grade = "黄金三角稍小"
                elif golden_angle_thresholds[2] <= golden_angle < golden_angle_thresholds[3]:
                    golden_angle_grade = "黄金三角标准"
                elif golden_angle_thresholds[3] <= golden_angle < golden_angle_thresholds[4]:
                    golden_angle_grade = "黄金三角稍大"
                elif golden_angle_thresholds[4] <= golden_angle < golden_angle_thresholds[5]:
                    golden_angle_grade = "黄金三角大"
                else:
                    golden_angle_grade = "黄金三角超大"
                print(f"黄金三角度数评价: {golden_angle_grade}")

                # 5. 眉毛
                print(f"\n● 眉毛")
                # 右眉(画面左)作为示例
                eyebrow_h = abs(landmarks[105].y * img_h - landmarks[55].y * img_h) * factor  # 眉毛高度
                eyebrow_w = get_distance(landmarks[70], landmarks[107], img_w, img_h) * factor  # 眉毛宽度
                eyebrow_thick = abs(landmarks[105].y * img_h - landmarks[223].y * img_h) * factor  # 眉毛粗细
                eyebrow_tilt = get_twopoint_angle(landmarks[105], landmarks[107], img_w, img_h)  # 挑度
                eyebrow_curvature = get_twopoint_angle(landmarks[46], landmarks[55], img_w, img_h)  # 弯度
                print(f"眉毛高度: {eyebrow_h:.2f} mm")
                print(f"眉毛长度: {eyebrow_w:.2f} mm")
                print(f"眉毛粗细: {eyebrow_thick:.2f} mm")
                print(f"眉毛挑度: {eyebrow_tilt:.2f} °")
                print(f"眉毛弯度: {eyebrow_curvature:.2f} °")
                # --- 眉毛宽度区间判断（表格数值：25.45、31.9236、36.7055、45.92、50.16、61.17）---
                eyebrow_w_thresholds = [25.45, 31.9236, 36.7055, 45.92, 50.16, 61.17]
                if eyebrow_w < eyebrow_w_thresholds[0]:
                    eyebrow_w_grade = "眉毛超短"
                elif eyebrow_w_thresholds[0] <= eyebrow_w < eyebrow_w_thresholds[1]:
                    eyebrow_w_grade = "眉毛短"
                elif eyebrow_w_thresholds[1] <= eyebrow_w < eyebrow_w_thresholds[2]:
                    eyebrow_w_grade = "眉毛稍短"
                elif eyebrow_w_thresholds[2] <= eyebrow_w < eyebrow_w_thresholds[3]:
                    eyebrow_w_grade = "眉毛标准"
                elif eyebrow_w_thresholds[3] <= eyebrow_w < eyebrow_w_thresholds[4]:
                    eyebrow_w_grade = "眉毛稍长"
                elif eyebrow_w_thresholds[4] <= eyebrow_w < eyebrow_w_thresholds[5]:
                    eyebrow_w_grade = "眉毛长"
                else:
                    eyebrow_w_grade = "眉毛超长"
                print(f"眉毛长度评价: {eyebrow_w_grade}")

                # 6. 眼睛
                print(f"\n● 眼睛")
                eye_height = get_distance(landmarks[159], landmarks[145], img_w, img_h) * factor  # 眼睛高度(上下睑)
                eye_width = get_distance(landmarks[33], landmarks[133], img_w, img_h) * factor  # 眼睛宽度
                inner_eye_angle = get_angle(landmarks[157], landmarks[133], landmarks[154], img_w, img_h)  # 内眦角度
                print(f"眼睛宽度: {eye_height:.2f} mm")
                print(f"眼睛长度: {eye_width:.2f} mm")
                print(f"内眦角度数: {inner_eye_angle:.2f} °")
                # --- 眼睛宽度区间判断 ---
                eye_width_thresholds = [6.171, 9.8076, 10.36, 11.7712, 12.53, 14.12]
                if eye_height < eye_width_thresholds[0]:
                    eye_width_grade = "眼宽超小"
                elif eye_width_thresholds[0] <= eye_height < eye_width_thresholds[1]:
                    eye_width_grade = "眼宽小"
                elif eye_width_thresholds[1] <= eye_height < eye_width_thresholds[2]:
                    eye_width_grade = "眼宽稍小"
                elif eye_width_thresholds[2] <= eye_height < eye_width_thresholds[3]:
                    eye_width_grade = "眼宽标准"
                elif eye_width_thresholds[3] <= eye_height < eye_width_thresholds[4]:
                    eye_width_grade = "眼宽稍大"
                elif eye_width_thresholds[4] <= eye_height < eye_width_thresholds[5]:
                    eye_width_grade = "眼宽大"
                else:
                    eye_width_grade = "眼宽超大"

                # 输出评价结果
                print(f"眼睛长度评价: {eye_grade}")
                print(f"眼睛宽度评价: {eye_width_grade}")

                # 7. 鼻子
                print(f"\n● 鼻子")
                nose_ala_width = get_distance(landmarks[129], landmarks[358], img_w, img_h) * factor
                print(f"鼻翼宽度: {nose_ala_width:.2f} mm")
                # --- 鼻翼宽度区间判断（表格数值：35.558、37.4328、38.73、40.96、42.58、45.172）---
                nose_ala_thresholds = [35.558, 37.4328, 38.73, 40.96, 42.58, 45.172]
                if nose_ala_width < nose_ala_thresholds[0]:
                    nose_ala_grade = "鼻翼超小"
                elif nose_ala_thresholds[0] <= nose_ala_width < nose_ala_thresholds[1]:
                    nose_ala_grade = "鼻翼小"
                elif nose_ala_thresholds[1] <= nose_ala_width < nose_ala_thresholds[2]:
                    nose_ala_grade = "鼻翼稍小"
                elif nose_ala_thresholds[2] <= nose_ala_width < nose_ala_thresholds[3]:
                    nose_ala_grade = "鼻翼标准"
                elif nose_ala_thresholds[3] <= nose_ala_width < nose_ala_thresholds[4]:
                    nose_ala_grade = "鼻翼稍大"
                elif nose_ala_thresholds[4] <= nose_ala_width < nose_ala_thresholds[5]:
                    nose_ala_grade = "鼻翼大"
                else:
                    nose_ala_grade = "鼻翼超大"
                print(f"鼻翼宽度评价: {nose_ala_grade}")

                # 8. 嘴唇
                print(f"\n● 嘴唇")
                lip_h = abs(landmarks[37].y * img_h - landmarks[17].y * img_h) * factor
                # 嘴唇宽度
                lip_w = get_distance(landmarks[61], landmarks[291], img_w, img_h) * factor
                # 厚度
                lip_up_thick = abs(landmarks[37].y * img_h - landmarks[13].y * img_h) * factor
                lip_down_thick = get_distance(landmarks[14], landmarks[17], img_w, img_h) * factor
                lip_thick = (lip_up_thick + lip_down_thick) / 2
                # 嘴角点(291)与唇中内缘点(13)
                mouth_angle = 270 - get_twopoint_angle(landmarks[308], landmarks[291], img_w, img_h)
                print(f"嘴唇高度: {lip_h:.2f} mm")
                print(f"嘴唇长度: {lip_w:.2f} mm")
                print(f"嘴唇厚度: {lip_thick:.2f} mm")
                print(f"嘴角弯曲度: {mouth_angle:.2f} °")
                # --- 嘴唇宽度区间判断（表格数值：37.961、42.7136、45.2485、51.92、58.044、66.494）---
                lip_w_thresholds = [37.961, 42.7136, 45.2485, 51.92, 58.044, 66.494]
                if lip_w < lip_w_thresholds[0]:
                    lip_w_grade = "嘴唇超短"
                elif lip_w_thresholds[0] <= lip_w < lip_w_thresholds[1]:
                    lip_w_grade = "嘴唇短"
                elif lip_w_thresholds[1] <= lip_w < lip_w_thresholds[2]:
                    lip_w_grade = "嘴唇稍短"
                elif lip_w_thresholds[2] <= lip_w < lip_w_thresholds[3]:
                    lip_w_grade = "嘴唇标准"
                elif lip_w_thresholds[3] <= lip_w < lip_w_thresholds[4]:
                    lip_w_grade = "嘴唇稍长"
                elif lip_w_thresholds[4] <= lip_w < lip_w_thresholds[5]:
                    lip_w_grade = "嘴唇长"
                else:
                    lip_w_grade = "嘴唇超长"
                print(f"嘴唇长度评价: {lip_w_grade}")

                mouth_angle_thresholds = [90.81, 93.1612, 98.33, 110.618, 120.42, 138.165]
                if mouth_angle < mouth_angle_thresholds[0]:
                    mouth_angle_grade = "嘴角超平"
                elif mouth_angle_thresholds[0] <= mouth_angle < mouth_angle_thresholds[1]:
                    mouth_angle_grade = "嘴角平"
                elif mouth_angle_thresholds[1] <= mouth_angle < mouth_angle_thresholds[2]:
                    mouth_angle_grade = "嘴角稍平"
                elif mouth_angle_thresholds[2] <= mouth_angle < mouth_angle_thresholds[3]:
                    mouth_angle_grade = "嘴角标准"
                elif mouth_angle_thresholds[3] <= mouth_angle < mouth_angle_thresholds[4]:
                    mouth_angle_grade = "嘴角稍翘"
                elif mouth_angle_thresholds[4] <= mouth_angle < mouth_angle_thresholds[5]:
                    mouth_angle_grade = "嘴角翘"
                else:
                    mouth_angle_grade = "嘴角超翘"
                print(f"嘴角弯曲度评价: {mouth_angle_grade}")

                # 9. 下巴
                print(f"\n● 下巴")
                # 下巴长度
                chin_len = get_distance(landmarks[17], landmarks[152], img_w, img_h) * factor
                # 下巴宽度
                chin_width = get_distance(landmarks[149], landmarks[378], img_w, img_h) * factor
                # 下巴角度
                chin_angle = get_angle(landmarks[149], landmarks[152], landmarks[378], img_w, img_h)
                print(f"下巴长度: {chin_len:.2f} mm")
                print(f"下巴宽度: {chin_width:.2f} mm")
                print(f"下巴角度: {chin_angle:.2f} °")
                # --- 下巴角度区间判断（表格数值：121.033、126.8828、129.9965、135.3212、138.6、145.847）---
                chin_angle_thresholds = [121.033, 126.8828, 129.9965, 135.3212, 138.6, 145.847]
                if chin_angle < chin_angle_thresholds[0]:
                    chin_angle_grade = "下巴超尖"
                elif chin_angle_thresholds[0] <= chin_angle < chin_angle_thresholds[1]:
                    chin_angle_grade = "下巴尖"
                elif chin_angle_thresholds[1] <= chin_angle < chin_angle_thresholds[2]:
                    chin_angle_grade = "下巴稍尖"
                elif chin_angle_thresholds[2] <= chin_angle < chin_angle_thresholds[3]:
                    chin_angle_grade = "下巴标准"
                elif chin_angle_thresholds[3] <= chin_angle < chin_angle_thresholds[4]:
                    chin_angle_grade = "下巴稍圆"
                elif chin_angle_thresholds[4] <= chin_angle < chin_angle_thresholds[5]:
                    chin_angle_grade = "下巴圆"
                else:
                    chin_angle_grade = "下巴超圆"
                print(f"下巴角度评价: {chin_angle_grade}")

                print(f"\n● 面部聚散度")
                face_density_ratio = get_face_density(landmarks, img_w, img_h)
                print(f"面部聚散度: {face_density_ratio:.2f}")
                face_density_thresholds = [0.28, 0.31, 0.32, 0.34, 0.35, 0.36]
                if chin_angle < chin_angle_thresholds[0]:
                    face_density_grade = "面部超聚拢"
                elif face_density_thresholds[0] <= face_density_ratio < face_density_thresholds[1]:
                    face_density_grade = "面部聚拢"
                elif face_density_thresholds[1] <= face_density_ratio < face_density_thresholds[2]:
                    face_density_grade = "面部稍聚拢"
                elif face_density_thresholds[2] <= face_density_ratio < face_density_thresholds[3]:
                    face_density_grade = "面部标准"
                elif face_density_thresholds[3] <= face_density_ratio < face_density_thresholds[4]:
                    face_density_grade = "面部稍分散"
                elif face_density_thresholds[4] <= face_density_ratio < face_density_thresholds[5]:
                    face_density_grade = "面部分散"
                else:
                    face_density_grade= "面部超分散"
                print(f"面部聚散度评价: {face_density_grade}")

                print(f"\n● 整体评价")
                print(f"整体: 这个人{mid_grade},{down_grade},{eye_dist_grade},{eye_grade},"
                      f"{face_len_grade},{zygomatic_grade},{mandible_angle_grade},"
                      f"{fwhr_grade},{golden_angle_grade},{eyebrow_w_grade},"
                      f"{eye_width_grade},{nose_ala_grade},{lip_w_grade},{mouth_angle_grade},"
                      f"{chin_angle_grade}, {face_density_grade}")

        else:
            print(f"❌ 图片 {image_name} 未检测到人脸")

    except Exception as e:
        print(f"❌ 处理图片 {os.path.basename(image_path)} 时出错：{str(e)}")


def get_valid_image_paths(input_path):
    """
    判断输入路径是文件还是文件夹，返回有效的图片路径列表
    :param input_path: 输入的文件/文件夹路径
    :return: 有效图片路径列表
    """
    valid_image_paths = []
    # 1. 判断路径是否存在
    if not os.path.exists(input_path):
        print(f"❌ 路径不存在：{input_path}")
        return valid_image_paths

    # 2. 如果是文件：判断是否为图片格式
    if os.path.isfile(input_path):
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png']:
            valid_image_paths.append(input_path)
        else:
            print(f"❌ 文件 {input_path} 不是支持的图片格式（仅支持jpg/jpeg/png）")
        return valid_image_paths

    # 3. 如果是文件夹：遍历所有图片格式文件（和原有逻辑一致）
    if os.path.isdir(input_path):
        # 匹配常见图片格式
        valid_image_paths = glob.glob(os.path.join(input_path, "*.[jJ][pP][gG]")) + \
                            glob.glob(os.path.join(input_path, "*.[jJ][pP][eE][gG]")) + \
                            glob.glob(os.path.join(input_path, "*.[pP][nN][gG]"))
        return valid_image_paths

    # 4. 既不是文件也不是文件夹
    print(f"❌ 路径 {input_path} 既不是文件也不是文件夹")
    return valid_image_paths


if __name__ == "__main__":
    # 1. 读取命令行参数
    if len(sys.argv) < 2:
        print("❌ 使用方法：python 脚本名.py <图片路径/文件夹路径>")
        sys.exit(1)
    input_path = sys.argv[1]
    # input_path ="C:\\Users\\Bruce\\Desktop\\1"

    # 2. 模型初始化
    model_path = "scripts/models/face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}，请下载后放在代码同目录下")
        exit(1)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, output_face_blendshapes=True)
    detector = vision.FaceLandmarker.create_from_options(options)

    # 3. 定义人脸关键点映射
    face_key_points = {
        # 三庭
        "额头发际线中心": 10,
        "双眉中心": 9,
        "鼻下点(人中顶部)": 2,
        "下巴尖": 152,
        # 五眼
        "右颧骨最外侧": 234,
        "右外眼角": 33,
        "右内眼角": 133,
        "左内眼角": 362,
        "左外眼角": 263,
        "左颧骨最外侧": 454,
        # 脸部
        "右颞部边缘": 127,
        "左颞部边缘": 356,
        "右下额边缘": 172,
        "左下额边缘": 397,
        "左耳耳际": 361,
        # 黄金三角
        "右眼瞳孔": 468,
        "左眼瞳孔": 473,
        "鼻尖": 1,
        # 眉毛
        "右眉外侧上部": 70,
        "右眉外侧下部": 46,
        "右眉最高点": 105,
        "右眉中部下": 223,
        "右眉内侧上部": 107,
        "右眉内侧下部": 55,
        # 眼睛
        "右眼上眼睑中心": 159,
        "右眼下眼睑中心": 145,
        "右眼靠内侧上方": 157,
        "右眼靠内侧下方": 154,
        # 鼻子
        "右鼻翼": 129,
        "左鼻翼": 358,
        # 嘴唇
        "右嘴角": 61,
        "左嘴角": 291,
        "上唇中点": 0,
        "上唇下部": 13,
        "下唇上部": 14,
        "下唇中点": 17,
        "上唇最高点": 37,
        "上唇下部偏左点": 308,
        # 下巴
        "下巴左侧": 378,
        "下巴右侧": 149
    }

    # 4. 获取有效图片路径列表
    image_paths = get_valid_image_paths(input_path)

    # 5. 处理图片
    if not image_paths:
        print(f"❌ 未找到可处理的有效图片")
    else:
        print(f"✅️ 共找到 {len(image_paths)} 张有效图片，开始处理...")
        number = 0
        for img_path in image_paths:
            number += 1
            process_single_image(img_path, detector, face_key_points, number)

    # 6. 释放检测器资源
    detector.close()
    print(f"\n==================================================")
    print(f"处理完成！")