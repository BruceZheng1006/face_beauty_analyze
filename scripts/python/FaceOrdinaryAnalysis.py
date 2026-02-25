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

# 所有基础数值的打印，对应第一个功能

# 强制标准输出/错误输出使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 普通人
MALE_STATS = {
    "top": {"mean": 0.1851695, "std": 0.0247877},  # 上庭占比
    "mid": {"mean": 0.4223434, "std": 0.0275991},  # 中庭占比
    "down": {"mean": 0.3924875, "std": 0.0437045},  # 下庭占比
    "dist_r_out_ratio": {"mean": 0.220658377, "std": 0.074476031},  # 右眼外侧留白占比
    "eye_length": {"mean": 26.32332781, "std": 2.21858849},  # 右眼长度(mm)
    "eye_dist_ratio": {"mean": 0.267333891, "std": 0.018747112},  # 内眼角间距占比
    "zygomatic": {"mean": 136.550995, "std": 7.668838016},  # 颧骨宽度(mm)
    "face_len": {"mean": 164.229096, "std": 14.1363018},  # 脸部长度(mm)
    "forehead_width": {"mean": 137.8560828, "std": 7.492267983},  # 颞部宽度(mm)
    "mandible_width": {"mean": 107.8112483, "std": 7.801552615},  # 下颌角宽度(mm)
    "mandible_angle": {"mean": 143.235952, "std": 6.431666412},  # 下颌角度数(°)
    "fwhr": {"mean": 1.626830811, "std": 0.155841504},  # 面部宽高比
    "golden_angle": {"mean": 73.86741225, "std": 11.01788869},  # 黄金三角度数
    "eyebrow_h": {"mean": 11.49860265, "std": 2.250228973},  # 眉毛高度(mm)
    "eyebrow_w": {"mean": 42.52375166, "std": 7.034906724},  # 眉毛长度(mm)
    "eyebrow_area": {"mean": 493.4991821, "std": 163.1901423},  # 眉毛大小(mm²)
    "eyebrow_thick": {"mean": 11.04552815, "std": 1.449998031},  # 眉毛粗细(mm)
    "eyebrow_tilt": {"mean": 6.078448675, "std": 3.893855245},  # 眉毛挑度(°)
    "eyebrow_curvature": {"mean": 3.715427152, "std": 3.767265264},  # 眉毛弯度(°)
    "eye_width": {"mean": 7.596733444, "std": 2.561378776},  # 眼睛宽度(mm)
    "eye_area": {"mean": 200.6629023, "std": 73.91103154},  # 眼睛面积(mm²)
    "inner_eye_angle": {"mean": 44.50140232, "std": 12.43391753},  # 内眦角度数(°)
    "relative_eye_area": {"mean": 0.010774953, "std": 0.004058049},  # 相对眼面积
    "eyelid_tilt": {"mean": 4.247018212, "std": 4.001509392},  # 眼睑裂倾斜度(°)
    "eyebrow_eye_length": {"mean": 12.4381455748553, "std": 3.75524116144724},  # 眉眼距(mm)
    "nose_ala": {"mean": 41.2943096, "std": 2.51309772},  # 鼻翼宽度(mm)
    "philtrum_length": {"mean": 15.23487417, "std": 2.592525147},  # 人中长度(mm)
    "lip_h": {"mean": 21.23677152, "std": 6.274081179},  # 嘴唇高度(mm)
    "lip_w": {"mean": 48.4487053, "std": 5.183063933},  # 嘴唇长度(mm)
    "lip_thick": {"mean": 9.283044702, "std": 1.673967682},  # 嘴唇厚度(mm)
    "lip_area": {"mean": 900.3601772, "std": 191.3521543},  # 嘴巴大小(mm²)
    "mouth_angle": {"mean": 97.89682781, "std": 6.803649571},  # 嘴角弯曲度(°)
    "chin_len": {"mean": 29.6091904, "std": 6.471615392},  # 下巴长度(mm)
    "chin_width": {"mean": 61.19029305, "std": 4.827839057},  # 下巴宽度(mm)
    "chin_angle": {"mean": 140.8392219, "std": 9.870431286},  # 下巴角度(°)
    "face_divergence": {"mean": 0.311201424, "std": 0.023956843},  # 面部聚散度
}

FEMALE_STATS = {
    "top": {"mean": 0.1981219, "std": 0.0223635},  # 上庭占比
    "mid": {"mean": 0.4411468, "std": 0.029467},  # 中庭占比
    "down": {"mean": 0.3607315, "std": 0.0439695},  # 下庭占比
    "dist_r_out_ratio": {"mean": 0.198029475, "std": 0.080718924},  # 右眼外侧留白占比
    "eye_length": {"mean": 26.69808784, "std": 2.400408629},  # 右眼长度(mm)
    "eye_dist_ratio": {"mean": 0.269955343, "std": 0.018818805},  # 内眼角间距占比
    "zygomatic": {"mean": 134.7811093, "std": 7.523259132},  # 颧骨宽度(mm)
    "face_len": {"mean": 161.1269725, "std": 13.64161952},  # 脸部长度(mm)
    "forehead_width": {"mean": 136.3885904, "std": 7.19038235},  # 颞部宽度(mm)
    "mandible_width": {"mean": 104.2011017, "std": 7.731770722},  # 下颌角宽度(mm)
    "mandible_angle": {"mean": 145.9947797, "std": 6.192613219},  # 下颌角度数(°)
    "fwhr": {"mean": 1.607160338, "std": 0.148666463},  # 面部宽高比
    "golden_angle": {"mean": 69.057639393, "std": 8.747980958},  # 黄金三角度数
    "eyebrow_h": {"mean": 12.49225777, "std": 2.043481661},  # 眉毛高度(mm)
    "eyebrow_w": {"mean": 41.43450045, "std": 7.551504583},  # 眉毛长度(mm)
    "eyebrow_area": {"mean": 518.433441, "std": 137.8575516},  # 眉毛大小(mm²)
    "eyebrow_thick": {"mean": 11.1751313, "std": 1.484078605},  # 眉毛粗细(mm)
    "eyebrow_tilt": {"mean": 8.044088439, "std": 4.252976134},  # 眉毛挑度(°)
    "eyebrow_curvature": {"mean": 3.611223966, "std": 3.522838122},  # 眉毛弯度(°)
    "eye_width": {"mean": 9.658602475, "std": 2.631877736},  # 眼睛宽度(mm)
    "eye_area": {"mean": 259.6842016, "std": 82.18488235},  # 眼睛面积(mm²)
    "inner_eye_angle": {"mean": 52.15427709, "std": 11.53557321},  # 内眦角度数(°)
    "relative_eye_area": {"mean": 0.014662185, "std": 0.005016915},  # 相对眼面积
    "eyelid_tilt": {"mean": 7.601595231, "std": 4.235200283},  # 眼睑裂倾斜度(°)
    "eyebrow_eye_length": {"mean": 11.4829977392615, "std": 2.49852274289319},  # 眉眼距(mm)
    "nose_ala": {"mean": 39.77429067, "std": 2.610021843},  # 鼻翼宽度(mm)
    "philtrum_length": {"mean": 13.4241307, "std": 2.580629796},  # 人中长度(mm)
    "lip_h": {"mean": 20.00716873, "std": 5.479511423},  # 嘴唇高度(mm)
    "lip_w": {"mean": 48.94343948, "std": 6.377723857},  # 嘴唇长度(mm)
    "lip_thick": {"mean": 8.518948083, "std": 1.506600678},  # 嘴唇厚度(mm)
    "lip_area": {"mean": 833.7871657, "std": 179.4436452},  # 嘴巴大小(mm²)
    "mouth_angle": {"mean": 102.300323, "std": 9.131610959},  # 嘴角弯曲度(°)
    "chin_len": {"mean": 26.34797465, "std": 6.174972035},  # 下巴长度(mm)
    "chin_width": {"mean": 56.92478267, "std": 4.997264542},  # 下巴宽度(mm)
    "chin_angle": {"mean": 135.4199321, "std": 8.627442248},  # 下巴角度(°)
    "face_divergence": {"mean": 0.316910625, "std": 0.020481753},  # 面部聚散度
}

def calculate_z_score(value, mean, std):
    """
    仅计算Z分数，无等级判断
    Z = (测量值 - 均值) / 标准差
    """
    # 异常防护：避免标准差为0导致除零错误
    if std <= 1e-6:
        return 0.0
    z_score = (value - mean) / std
    return round(z_score, 4)


def get_3d_face_fold_score(landmarks, face_length, zygomatic_width, mandible_width):
    """计算3D面部折叠度综合分数"""
    if zygomatic_width <= 0:
        return 0.0
    jaw_zygo_ratio = mandible_width / zygomatic_width
    len_zygo_ratio = face_length / zygomatic_width
    base_2d_score = (1 - jaw_zygo_ratio) * 0.5 + len_zygo_ratio * 0.2
    base_z = (landmarks[10].z + landmarks[152].z) / 2
    nose_z = landmarks[1].z
    depth_diff = abs(nose_z - base_z)
    depth_score = depth_diff
    final_score = base_2d_score + depth_score
    return round(final_score, 4)


def get_face_divergence(landmarks, img_w, img_h):
    """计算面部聚散度"""

    def to_pixel(idx):
        return [landmarks[idx].x * img_w, landmarks[idx].y * img_h]

    inner_indices = [8, 33, 61, 18, 291, 263]
    inner_pts = np.array([to_pixel(i) for i in inner_indices], dtype=np.int32)
    outer_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    outer_pts = np.array([to_pixel(i) for i in outer_indices], dtype=np.int32)
    area_inner = cv2.contourArea(inner_pts)
    area_outer = cv2.contourArea(outer_pts)
    density_ratio = area_inner / area_outer if area_outer > 0 else 0
    return density_ratio


# --- 几何计算工具函数 ---
def get_distance(p1, p2, img_w, img_h):
    return math.sqrt(((p1.x - p2.x) * img_w) ** 2 + ((p1.y - p2.y) * img_h) ** 2)


def get_angle(p1, p2, p3, img_w, img_h):
    a = [p1.x * img_w, p1.y * img_h]
    b = [p2.x * img_w, p2.y * img_h]
    c = [p3.x * img_w, p3.y * img_h]
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)


def get_twopoint_angle(p1, p2, img_w, img_h):
    a = [p1.x * img_w, p1.y * img_h]
    b = [p2.x * img_w, p2.y * img_h]
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    return abs(ang) if abs(ang) <= 180 else 360 - abs(ang)


# 新增：计算两点连线的倾斜度（斜率角度）
def get_line_tilt(p1, p2, img_w, img_h):
    """计算两点连线的倾斜角度（相对于水平线）"""
    a = [p1.x * img_w, p1.y * img_h]
    b = [p2.x * img_w, p2.y * img_h]
    # 计算斜率
    if (b[0] - a[0]) == 0:  # 垂直线
        return 90.0
    slope = (b[1] - a[1]) / (b[0] - a[0])
    # 计算倾斜角度
    tilt_angle = math.degrees(math.atan(slope))
    return tilt_angle


# --- 封装单张图片处理函数 ---
def process_single_image(image_path, detector, number, gender):
    try:
        # 获取对应性别的统计参数
        stats = FEMALE_STATS if gender.lower() == 'female' else MALE_STATS
        image_name = os.path.basename(image_path)
        print(f"\n==================================================")
        print(f"正在处理图片：{image_name}")
        print(f"==================================================")

        # 加载图片并检测
        image = mp.Image.create_from_file(image_path)
        img_h, img_w = image.height, image.width
        result = detector.detect(image)

        # 初始化Z分数列表，用于统计绝对值大于1的个数
        z_scores_list = []

        if result.face_landmarks:
            for face_idx, landmarks in enumerate(result.face_landmarks):
                # 比例尺计算
                px_ipd = get_distance(landmarks[468], landmarks[473], img_w, img_h)
                factor = 63.0 / px_ipd if px_ipd > 0 else 1.0
                print(f"\n========== 深度美学测量 (单位: mm) ==========")

                # 1. 三庭
                print(f"\n● 三庭")
                top_h = get_distance(landmarks[10], landmarks[9], img_w, img_h) * factor
                mid_h = get_distance(landmarks[9], landmarks[2], img_w, img_h) * factor
                bot_h = get_distance(landmarks[2], landmarks[152], img_w, img_h) * factor
                total_h = top_h + mid_h + bot_h
                top_ratio = top_h / total_h if total_h > 0 else 0
                mid_ratio = mid_h / total_h if total_h > 0 else 0
                down_ratio = bot_h / total_h if total_h > 0 else 0

                print(f"三庭比例: {top_ratio:.2f}:{mid_ratio:.2f}:{down_ratio:.2f}")
                print(f"上庭长度: {top_h:.2f} mm")
                print(f"上庭占比: {top_ratio:.2f}")
                print(f"中庭长度: {mid_h:.2f} mm")
                print(f"中庭占比: {mid_ratio:.2f}")
                print(f"下庭长度: {bot_h:.2f} mm")
                print(f"下庭占比: {down_ratio:.2f}")
                # 仅打印Z分数并加入列表
                top_z = calculate_z_score(top_ratio, stats["top"]["mean"], stats["top"]["std"])
                z_scores_list.append(top_z)
                print(f"上庭尺寸评价（Z-score）: {top_z}")
                mid_z = calculate_z_score(mid_ratio, stats["mid"]["mean"], stats["mid"]["std"])
                z_scores_list.append(mid_z)
                print(f"中庭尺寸评价（Z-score）: {mid_z}")
                down_z = calculate_z_score(down_ratio, stats["down"]["mean"], stats["down"]["std"])
                z_scores_list.append(down_z)
                print(f"下庭尺寸评价（Z-score）: {down_z}")

                # 2. 五眼
                print(f"\n● 五眼")
                # 颧骨宽
                zygomatic_width = get_distance(landmarks[234], landmarks[454], img_w, img_h) * factor  # 颧骨宽度
                dist_r_out = get_distance(landmarks[234], landmarks[33], img_w, img_h) * factor
                dist_r_out_ratio = dist_r_out / zygomatic_width if zygomatic_width > 0 else 0
                dist_r_eye = get_distance(landmarks[33], landmarks[133], img_w, img_h) * factor
                dist_mid = get_distance(landmarks[133], landmarks[362], img_w, img_h) * factor
                dist_mid_ratio = dist_mid / zygomatic_width if zygomatic_width > 0 else 0
                dist_l_eye = get_distance(landmarks[362], landmarks[263], img_w, img_h) * factor
                dist_l_out = get_distance(landmarks[263], landmarks[454], img_w, img_h) * factor
                print(
                    f"五眼比例: {dist_r_out / dist_mid:.2f}:{dist_r_eye / dist_mid:.2f}:1.00:{dist_l_eye / dist_mid:.2f}:{dist_l_out / dist_mid:.2f}")
                print(f"右眼外侧留白距离: {dist_r_out:.2f} mm")
                print(f"右眼外侧留白距离占比: {dist_r_out_ratio:.2f}")
                print(f"右眼长度: {dist_r_eye:.2f} mm")
                print(f"内眼角间距: {dist_mid:.2f} mm")  # 通常作为基准1
                print(f"内眼角间距占比: {dist_mid_ratio:.2f}")
                print(f"左眼长度: {dist_l_eye:.2f} mm")
                print(f"左眼外侧留白距离: {dist_l_out:.2f} mm")
                print(f"左眼外侧留白距离占比: {dist_l_out / zygomatic_width:.2f}")
                dist_r_out_ratio_z = calculate_z_score(dist_r_out_ratio, stats["dist_r_out_ratio"]["mean"],
                                                       stats["dist_r_out_ratio"]["std"])
                z_scores_list.append(dist_r_out_ratio_z)
                print(f"右眼外侧留白评价（Z-score）: {dist_r_out_ratio_z}")
                eye_dist_ratio_z = calculate_z_score(dist_mid_ratio, stats["eye_dist_ratio"]["mean"],
                                                     stats["eye_dist_ratio"]["std"])
                z_scores_list.append(eye_dist_ratio_z)
                print(f"内眼角间距评价（Z-score）: {eye_dist_ratio_z}")

                # 3. 脸部
                print(f"\n● 脸部")
                face_length = get_distance(landmarks[10], landmarks[152], img_w, img_h) * factor
                forehead_width = get_distance(landmarks[127], landmarks[356], img_w, img_h) * factor  # 颞部宽度
                zygomatic_width = get_distance(landmarks[234], landmarks[454], img_w, img_h) * factor  # 颧骨宽度
                mandible_width = get_distance(landmarks[172], landmarks[397], img_w, img_h) * factor  # 下颌角宽度
                mandible_angle = get_angle(landmarks[361], landmarks[397], landmarks[152], img_w, img_h)  # 下颌角度数
                upface_height = get_distance(landmarks[9], landmarks[0], img_w, img_h) * factor  # 上脸高度
                # fWHR:颧骨宽度 / 上脸高度(眉毛中心点到上唇上缘)
                fwhr = zygomatic_width / upface_height if upface_height > 0 else 0
                print(f"脸部长度: {face_length:.2f} mm")
                print(f"颞部宽度: {forehead_width:.2f} mm")
                print(f"颧骨宽度: {zygomatic_width:.2f} mm")
                print(f"下颌角宽度: {mandible_width:.2f} mm")
                print(f"下颌角度数: {mandible_angle:.2f} °")
                print(
                    f"颞宽:颧宽:下颌宽: {forehead_width / zygomatic_width:.2f}:1.00:{mandible_width / zygomatic_width:.2f}")
                print(f"fWHR(面部宽高比): {fwhr:.2f}")

                face_len_z = calculate_z_score(face_length, stats["face_len"]["mean"], stats["face_len"]["std"])
                z_scores_list.append(face_len_z)
                forehead_z = calculate_z_score(forehead_width, stats["forehead_width"]["mean"],
                                               stats["forehead_width"]["std"])
                z_scores_list.append(forehead_z)
                zygomatic_z = calculate_z_score(zygomatic_width, stats["zygomatic"]["mean"], stats["zygomatic"]["std"])
                z_scores_list.append(zygomatic_z)
                mandible_z = calculate_z_score(mandible_width, stats["mandible_width"]["mean"],
                                               stats["mandible_width"]["std"])
                z_scores_list.append(mandible_z)
                mandible_angle_z = calculate_z_score(mandible_angle, stats["mandible_angle"]["mean"],
                                                     stats["mandible_angle"]["std"])
                z_scores_list.append(mandible_angle_z)
                fwhr_z = calculate_z_score(fwhr, stats["fwhr"]["mean"], stats["fwhr"]["std"])
                z_scores_list.append(fwhr_z)
                print(f"脸部长度评价（Z-score）: {face_len_z}")
                print(f"颞部（太阳穴）宽度评价（Z-score）: {forehead_z}")
                print(f"颧骨宽度评价（Z-score）: {zygomatic_z}")
                print(f"下颌角宽度评价（Z-score）: {mandible_z}")
                print(f"下颌角度数评价（Z-score）: {mandible_angle_z}")
                print(f"面部宽高比评价（Z-score）: {fwhr_z}（越大越方）")

                # 4. 黄金三角
                print(f"\n● 黄金三角")
                golden_angle = get_angle(landmarks[468], landmarks[1], landmarks[473], img_w, img_h)
                print(f"黄金三角度数: {golden_angle:.2f} °")
                golden_angle_z = calculate_z_score(golden_angle, stats["golden_angle"]["mean"],
                                                   stats["golden_angle"]["std"])
                z_scores_list.append(golden_angle_z)
                print(f"黄金三角度数评价（Z-score）: {golden_angle_z}")

                # 5. 眉毛
                print(f"\n● 眉毛")
                eyebrow_h = abs(landmarks[105].y * img_h - landmarks[55].y * img_h) * factor  # 眉毛高度
                eyebrow_w = get_distance(landmarks[70], landmarks[107], img_w, img_h) * factor  # 眉毛宽度
                eyebrow_area = eyebrow_h * eyebrow_w
                eyebrow_thick = abs(landmarks[105].y * img_h - landmarks[223].y * img_h) * factor  # 眉毛粗细
                eyebrow_tilt = get_twopoint_angle(landmarks[105], landmarks[107], img_w, img_h)  # 挑度
                eyebrow_curvature = get_twopoint_angle(landmarks[46], landmarks[55], img_w, img_h)  # 弯度

                # 输出基础数值 + 每个特征的Z分数评价
                eyebrow_h_z = calculate_z_score(eyebrow_h, stats["eyebrow_h"]["mean"], stats["eyebrow_h"]["std"])
                z_scores_list.append(eyebrow_h_z)

                eyebrow_w_z = calculate_z_score(eyebrow_w, stats["eyebrow_w"]["mean"], stats["eyebrow_w"]["std"])
                z_scores_list.append(eyebrow_w_z)

                eyebrow_area_z = calculate_z_score(eyebrow_area, stats["eyebrow_area"]["mean"],
                                                   stats["eyebrow_area"]["std"])
                z_scores_list.append(eyebrow_area_z)

                eyebrow_thick_z = calculate_z_score(eyebrow_thick, stats["eyebrow_thick"]["mean"],
                                                    stats["eyebrow_thick"]["std"])
                z_scores_list.append(eyebrow_thick_z)

                eyebrow_tilt_z = calculate_z_score(eyebrow_tilt, stats["eyebrow_tilt"]["mean"],
                                                   stats["eyebrow_tilt"]["std"])
                z_scores_list.append(eyebrow_tilt_z)

                eyebrow_curvature_z = calculate_z_score(eyebrow_curvature, stats["eyebrow_curvature"]["mean"],
                                                        stats["eyebrow_curvature"]["std"])
                z_scores_list.append(eyebrow_curvature_z)

                print(f"眉毛宽度: {eyebrow_h:.2f} mm")
                print(f"眉毛长度: {eyebrow_w:.2f} mm")
                print(f"眉毛大小: {eyebrow_area:.2f} mm²")
                print(f"眉毛粗细: {eyebrow_thick:.2f} mm")
                print(f"眉毛挑度: {eyebrow_tilt:.2f} °")
                print(f"眉毛弯度: {eyebrow_curvature:.2f} °")

                print(f"眉毛宽度评价（Z-score）: {eyebrow_h_z}")
                print(f"眉毛长度评价（Z-score）: {eyebrow_w_z}")
                print(f"眉毛大小评价（Z-score）: {eyebrow_area_z}")
                print(f"眉毛粗细评价（Z-score）: {eyebrow_thick_z}")
                print(f"眉毛挑度评价（Z-score）: {eyebrow_tilt_z}")
                print(f"眉毛弯度评价（Z-score）: {eyebrow_curvature_z}")

                # 6. 眼睛
                print(f"\n● 眼睛")
                eye_width = get_distance(landmarks[159], landmarks[145], img_w, img_h) * factor
                eye_length = get_distance(landmarks[33], landmarks[133], img_w, img_h) * factor
                inner_eye_angle = get_angle(landmarks[157], landmarks[133], landmarks[154], img_w, img_h)  # 内眦角度
                eye_area = eye_length * eye_width

                # 计算面部面积（使用面部外轮廓关键点）
                face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                face_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in face_contour_indices],
                                    dtype=np.int32)
                face_area_px = cv2.contourArea(face_pts)  # 像素面积
                face_area = face_area_px * (factor ** 2)  # 转换为实际面积（mm²）
                relative_eye_area = eye_area / face_area if face_area > 0 else 0  # 相对眼面积

                eyelid_tilt = get_line_tilt(landmarks[33], landmarks[133], img_w, img_h)  # 右眼眼睑裂倾斜度

                eye_length_z = calculate_z_score(eye_length, stats["eye_length"]["mean"], stats["eye_length"]["std"])
                z_scores_list.append(eye_length_z)

                eye_width_z = calculate_z_score(eye_width, stats["eye_width"]["mean"], stats["eye_width"]["std"])
                z_scores_list.append(eye_width_z)

                eye_area_z = calculate_z_score(eye_area, stats["eye_area"]["mean"], stats["eye_area"]["std"])
                z_scores_list.append(eye_area_z)

                inner_eye_angle_z = calculate_z_score(inner_eye_angle, stats["inner_eye_angle"]["mean"],
                                                      stats["inner_eye_angle"]["std"])
                z_scores_list.append(inner_eye_angle_z)
                relative_eye_area_z = calculate_z_score(relative_eye_area, stats["relative_eye_area"]["mean"],
                                                        stats["relative_eye_area"]["std"])
                z_scores_list.append(relative_eye_area_z)
                eyelid_tilt_z = calculate_z_score(eyelid_tilt, stats["eyelid_tilt"]["mean"],
                                                  stats["eyelid_tilt"]["std"])
                z_scores_list.append(eyelid_tilt_z)

                # 眉眼距
                eyebrow_eye_length = get_distance(landmarks[223], landmarks[159], img_w, img_h) * factor
                eyebrow_eye_length_z = calculate_z_score(eyebrow_eye_length, stats["eyebrow_eye_length"]["mean"],
                                                         stats["eyebrow_eye_length"]["std"])
                z_scores_list.append(eyebrow_eye_length_z)

                print(f"眼睛长度: {eye_length:.2f} mm")
                print(f"眼睛宽度: {eye_width:.2f} mm")
                print(f"眼睛大小: {eye_area:.2f} mm²")
                print(f"内眦角度数: {inner_eye_angle:.2f} °")
                print(f"相对眼面积: {relative_eye_area:.6f}")
                print(f"右眼眼睑裂倾斜度: {eyelid_tilt:.2f} °")
                print(f"眉眼距: {eyebrow_eye_length:.2f} mm")

                print(f"眼睛长度评价（Z-score）: {eye_length_z}")
                print(f"眼睛宽度评价（Z-score）: {eye_width_z}")
                print(f"眼睛大小评价（Z-score）: {eye_area_z}")
                print(f"内眦角度数评价（Z-score）: {inner_eye_angle_z}")
                print(f"相对眼面积评价（Z-score）: {relative_eye_area_z}（越大眼睛相对面部越大）")
                print(f"眼睑裂倾斜度评价（Z-score）: {eyelid_tilt_z}（正值向上倾斜，负值向下倾斜）")
                print(f"眉眼距评价（Z-score）: {eyebrow_eye_length_z}")

                # 7. 鼻子
                print(f"\n● 鼻子")
                nose_ala_width = get_distance(landmarks[129], landmarks[358], img_w, img_h) * factor
                philtrum_length = get_distance(landmarks[2], landmarks[0], img_w, img_h) * factor

                nose_ala_z = calculate_z_score(nose_ala_width, stats["nose_ala"]["mean"], stats["nose_ala"]["std"])
                z_scores_list.append(nose_ala_z)

                print(f"鼻翼宽度: {nose_ala_width:.2f} mm")
                print(f"人中长度: {philtrum_length:.2f} mm")
                philtrum_z = calculate_z_score(philtrum_length, stats["philtrum_length"]["mean"],
                                               stats["philtrum_length"]["std"])
                z_scores_list.append(philtrum_z)

                print(f"鼻翼宽度评价（Z-score）: {nose_ala_z}")
                print(f"人中长度评价（Z-score）: {philtrum_z}")

                # 8. 嘴唇
                print(f"\n● 嘴唇")
                lip_h = abs(landmarks[37].y * img_h - landmarks[17].y * img_h) * factor
                lip_w = get_distance(landmarks[61], landmarks[291], img_w, img_h) * factor
                lip_up_thick = abs(landmarks[37].y * img_h - landmarks[13].y * img_h) * factor
                lip_down_thick = get_distance(landmarks[14], landmarks[17], img_w, img_h) * factor
                lip_thick = (lip_up_thick + lip_down_thick) / 2
                lip_up_area = lip_up_thick * lip_w
                lip_down_area = lip_down_thick * lip_w
                lip_area = lip_up_area + lip_down_area
                mouth_angle = 270 - get_twopoint_angle(landmarks[291], landmarks[308], img_w, img_h)
                lip_h_z = calculate_z_score(lip_h, stats["lip_h"]["mean"], stats["lip_h"]["std"])
                z_scores_list.append(lip_h_z)


                lip_w_z = calculate_z_score(lip_w, stats["lip_w"]["mean"], stats["lip_w"]["std"])
                z_scores_list.append(lip_w_z)

                lip_thick_z = calculate_z_score(lip_thick, stats["lip_thick"]["mean"], stats["lip_thick"]["std"])
                z_scores_list.append(lip_thick_z)

                lip_area_z = calculate_z_score(lip_area, stats["lip_area"]["mean"], stats["lip_area"]["std"])
                z_scores_list.append(lip_area_z)


                mouth_angle_z = calculate_z_score(mouth_angle, stats["mouth_angle"]["mean"],
                                                  stats["mouth_angle"]["std"])
                z_scores_list.append(mouth_angle_z)

                print(f"嘴唇高度: {lip_h:.2f} mm")
                print(f"嘴唇长度: {lip_w:.2f} mm")
                print(f"上唇厚度: {lip_up_thick:.2f} mm")
                print(f"下唇厚度: {lip_down_thick:.2f} mm")
                print(f"嘴唇厚度: {lip_thick:.2f} mm")
                print(f"嘴巴大小: {lip_area:.2f} mm²")
                print(f"嘴角弯曲度: {mouth_angle:.2f} °")

                print(f"嘴唇高度评价（Z-score）: {lip_h_z}")
                print(f"嘴唇长度评价（Z-score）: {lip_w_z}")
                print(f"嘴唇厚度评价（Z-score）: {lip_thick_z}")
                print(f"嘴巴大小评价（Z-score）: {lip_area_z}")
                print(f"嘴角弯曲度评价（Z-score）: {mouth_angle_z}")

                # 9. 下巴
                print(f"\n● 下巴")
                # 下巴长度
                chin_len = get_distance(landmarks[17], landmarks[152], img_w, img_h) * factor
                # 下巴宽度
                chin_width = get_distance(landmarks[149], landmarks[378], img_w, img_h) * factor
                # 下巴角度
                chin_angle = get_angle(landmarks[149], landmarks[152], landmarks[378], img_w, img_h)

                # 输出基础数值 + 每个特征的Z分数评价
                print(f"下巴长度: {chin_len:.2f} mm")
                chin_len_z = calculate_z_score(chin_len, stats["chin_len"]["mean"], stats["chin_len"]["std"])
                z_scores_list.append(chin_len_z)

                print(f"下巴宽度: {chin_width:.2f} mm")
                chin_width_z = calculate_z_score(chin_width, stats["chin_width"]["mean"], stats["chin_width"]["std"])
                z_scores_list.append(chin_width_z)

                print(f"下巴角度: {chin_angle:.2f} °")
                chin_angle_z = calculate_z_score(chin_angle, stats["chin_angle"]["mean"], stats["chin_angle"]["std"])
                z_scores_list.append(chin_angle_z)

                print(f"下巴长度评价（Z-score）: {chin_len_z}")
                print(f"下巴宽度评价（Z-score）: {chin_width_z}")
                print(f"下巴角度评价（Z-score）: {chin_angle_z}(越正越圆)")

                # 10. 面部聚散度
                print(f"\n● 面部聚散度")
                face_divergence_ratio = get_face_divergence(landmarks, img_w, img_h)
                print(f"面部聚散度: {face_divergence_ratio:.4f}")
                face_divergence_z = calculate_z_score(face_divergence_ratio, stats["face_divergence"]["mean"],
                                                      stats["face_divergence"]["std"])
                z_scores_list.append(face_divergence_z)
                print(f"面部聚散度评价（Z-score）: {face_divergence_z}(越正越分散)")

                print(f"\n● 照片类型")
                # 计算照片总面积（像素）
                photo_area_px = img_w * img_h
                # 脸部像素面积（已在眼睛部分计算）
                # 脸部面积 / 照片面积 比值
                face_photo_ratio = face_area_px / photo_area_px if photo_area_px > 0 else 0
                print(f"脸部像素面积: {face_area_px:.2f} 像素")
                print(f"照片像素面积: {photo_area_px:.2f} 像素")
                print(f"脸部/照片面积比: {face_photo_ratio:.4f}")
                # 照片类型判断（可根据业务调整阈值）
                if face_photo_ratio > 0.16:
                    photo_type = "特写（大头照）"
                elif face_photo_ratio > 0.04:
                    photo_type = "半身照"
                else:
                    photo_type = "全身照/远景照"
                print(f"照片类型判定: {photo_type}")

                print(f"\n● 总体评价")
                # 统计Z分数绝对值大于1的个数
                total_z_gt_1 = sum(1 for z in z_scores_list if abs(z) > 1)
                print(f"Z分数绝对值大于1的特征数: {total_z_gt_1}")

        else:
            print(f"❌ 图片 {image_name} 未检测到人脸")

    except Exception as e:
        print(f"❌ 处理图片 {os.path.basename(image_path)} 时出错：{str(e)}")
        # 打印异常堆栈信息，便于调试
        import traceback
        traceback.print_exc()


def get_valid_image_paths(input_path):
    """获取有效图片路径列表"""
    valid_image_paths = []
    if not os.path.exists(input_path):
        print(f"❌ 路径不存在：{input_path}")
        return valid_image_paths
    if os.path.isfile(input_path):
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png']:
            valid_image_paths.append(input_path)
        else:
            print(f"❌ 文件格式不支持，仅支持jpg/jpeg/png")
        return valid_image_paths
    if os.path.isdir(input_path):
        valid_image_paths = glob.glob(os.path.join(input_path, "*.[jJ][pP][gG]")) + \
                            glob.glob(os.path.join(input_path, "*.[jJ][pP][eE][gG]")) + \
                            glob.glob(os.path.join(input_path, "*.[pP][nN][gG]"))
    return valid_image_paths


if __name__ == "__main__":
    # 命令行参数校验
    if len(sys.argv) < 3:
        print("❌ 使用方法：python 脚本名.py <图片路径/文件夹路径> <性别(male/female)>")
        sys.exit(1)
    input_path = sys.argv[1]
    gender = sys.argv[2].lower()
    # input_path = "C:\\Users\\Bruce\\Desktop\\912.png"
    # gender = "male"
    if gender not in ['male', 'female']:
        print("❌ 性别参数错误，请使用 male 或 female")
        sys.exit(1)

    # 模型初始化
    model_path = "scripts/models/face_landmarker.task"
    # model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        exit(1)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, output_face_blendshapes=True)
    detector = vision.FaceLandmarker.create_from_options(options)

    # 批量处理图片
    image_paths = get_valid_image_paths(input_path)
    if not image_paths:
        print(f"❌ 未找到可处理的有效图片")
    else:
        print(f"✅️ 共找到 {len(image_paths)} 张有效图片，开始处理...")
        for idx, img_path in enumerate(image_paths, 1):
            process_single_image(img_path, detector, idx, gender)

    detector.close()
    print(f"\n==================================================")
    print(f"处理完成！")