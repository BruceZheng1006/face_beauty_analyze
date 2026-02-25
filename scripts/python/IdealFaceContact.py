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
import json

# 对应第三个功能，给出一张基准照片，得出距离该照片的长度

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
    """计算Z分数"""
    if std <= 1e-6:
        return 0.0
    z_score = (value - mean) / std
    return round(z_score, 4)


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


def get_line_tilt(p1, p2, img_w, img_h):
    """计算两点连线的倾斜角度"""
    a = [p1.x * img_w, p1.y * img_h]
    b = [p2.x * img_w, p2.y * img_h]
    if (b[0] - a[0]) == 0:
        return 90.0
    slope = (b[1] - a[1]) / (b[0] - a[0])
    tilt_angle = math.degrees(math.atan(slope))
    return tilt_angle


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
    density_ratio = area_inner / area_outer
    return density_ratio


def extract_face_features(image_path, detector, gender):
    """提取人脸特征并计算Z分数"""
    stats = FEMALE_STATS if gender.lower() == 'female' else MALE_STATS

    try:
        # 加载图片
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        img_h, img_w = img.shape[:2]
        result = detector.detect(mp_image)

        if not result.face_landmarks:
            return None, "未检测到人脸"

        landmarks = result.face_landmarks[0]

        # 比例尺计算
        px_ipd = get_distance(landmarks[468], landmarks[473], img_w, img_h)
        factor = 63.0 / px_ipd

        # 提取所有特征值
        features = {}

        # 三庭
        top_h = get_distance(landmarks[10], landmarks[9], img_w, img_h) * factor
        mid_h = get_distance(landmarks[9], landmarks[2], img_w, img_h) * factor
        bot_h = get_distance(landmarks[2], landmarks[152], img_w, img_h) * factor
        total_h = top_h + mid_h + bot_h
        top_ratio = top_h / total_h
        mid_ratio = mid_h / total_h
        down_ratio = bot_h / total_h

        # features["top"] = top_ratio
        features["mid"] = mid_ratio
        features["down"] = down_ratio

        # 五眼
        zygomatic_width = get_distance(landmarks[234], landmarks[454], img_w, img_h) * factor
        dist_r_out = get_distance(landmarks[234], landmarks[33], img_w, img_h) * factor
        dist_r_out_ratio = dist_r_out / zygomatic_width
        dist_mid = get_distance(landmarks[133], landmarks[362], img_w, img_h) * factor
        dist_mid_ratio = dist_mid / zygomatic_width

        features["dist_r_out_ratio"] = dist_r_out_ratio
        features["eye_dist_ratio"] = dist_mid_ratio
        features["eye_length"] = get_distance(landmarks[33], landmarks[133], img_w, img_h) * factor

        # 脸部
        features["face_len"] = get_distance(landmarks[10], landmarks[152], img_w, img_h) * factor
        features["forehead_width"] = get_distance(landmarks[127], landmarks[356], img_w, img_h) * factor
        features["zygomatic"] = get_distance(landmarks[234], landmarks[454], img_w, img_h) * factor
        features["mandible_width"] = get_distance(landmarks[172], landmarks[397], img_w, img_h) * factor
        features["mandible_angle"] = get_angle(landmarks[361], landmarks[397], landmarks[152], img_w, img_h)
        upface_height = get_distance(landmarks[9], landmarks[0], img_w, img_h) * factor
        features["fwhr"] = features["zygomatic"] / upface_height

        # 黄金三角
        features["golden_angle"] = get_angle(landmarks[468], landmarks[1], landmarks[473], img_w, img_h)

        #         # 眉毛
        #         features["eyebrow_h"] = abs(landmarks[105].y * img_h - landmarks[55].y * img_h) * factor
        #         features["eyebrow_w"] = get_distance(landmarks[70], landmarks[107], img_w, img_h) * factor
        #         features["eyebrow_area"] = features["eyebrow_h"] * features["eyebrow_w"]
        #         features["eyebrow_thick"] = abs(landmarks[105].y * img_h - landmarks[223].y * img_h) * factor
        #         features["eyebrow_tilt"] = get_twopoint_angle(landmarks[105], landmarks[107], img_w, img_h)
        #         features["eyebrow_curvature"] = get_twopoint_angle(landmarks[46], landmarks[55], img_w, img_h)
        #
        # 眼睛
        features["eye_width"] = get_distance(landmarks[159], landmarks[145], img_w, img_h) * factor
        eye_area = features["eye_length"] * features["eye_width"]
        features["eye_area"] = eye_area

        # 计算面部面积用于相对眼面积
        face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        face_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in face_contour_indices],
                            dtype=np.int32)
        face_area = cv2.contourArea(face_pts) * (factor ** 2)
        features["relative_eye_area"] = eye_area / face_area
        # features["inner_eye_angle"] = get_angle(landmarks[157], landmarks[133], landmarks[154], img_w, img_h)
        # features["eyelid_tilt"] = get_line_tilt(landmarks[33], landmarks[133], img_w, img_h)
        features["eyebrow_eye_length"] = get_distance(landmarks[223], landmarks[159], img_w, img_h) * factor

        # 鼻子
        features["nose_ala"] = get_distance(landmarks[129], landmarks[358], img_w, img_h) * factor
        features["philtrum_length"] = get_distance(landmarks[2], landmarks[0], img_w, img_h) * factor

        # 嘴唇
        features["lip_h"] = abs(landmarks[37].y * img_h - landmarks[17].y * img_h) * factor
        features["lip_w"] = get_distance(landmarks[61], landmarks[291], img_w, img_h) * factor
        lip_up_thick = abs(landmarks[37].y * img_h - landmarks[13].y * img_h) * factor
        lip_down_thick = get_distance(landmarks[14], landmarks[17], img_w, img_h) * factor
        lip_thick = (lip_up_thick + lip_down_thick) / 2
        features["lip_thick"] = lip_thick
        lip_up_area = lip_up_thick * features["lip_w"]
        lip_down_area = lip_down_thick * features["lip_w"]
        features["lip_area"] = lip_up_area + lip_down_area
        features["mouth_angle"] = 270 - get_twopoint_angle(landmarks[291], landmarks[308], img_w, img_h)

        # 下巴
        features["chin_len"] = get_distance(landmarks[17], landmarks[152], img_w, img_h) * factor
        features["chin_width"] = get_distance(landmarks[149], landmarks[378], img_w, img_h) * factor
        features["chin_angle"] = get_angle(landmarks[149], landmarks[152], landmarks[378], img_w, img_h)

        # 面部聚散度
        features["face_divergence"] = get_face_divergence(landmarks, img_w, img_h)

        # 计算Z分数
        z_scores = {}
        for feature_name, value in features.items():
            if feature_name in stats:
                z_scores[feature_name] = calculate_z_score(value, stats[feature_name]["mean"],
                                                           stats[feature_name]["std"])
            else:
                z_scores[feature_name] = 0.0

        return z_scores, "成功"

    except Exception as e:
        return None, f"处理失败: {str(e)}"


def beauty_face_score(gender, beauty_cols_name):
    beauty_face = {}
    if gender == "male":
        beauty_face = {
            "top": -0.0241,
            "mid": -0.1940,
            "down": 0.1362,
            "dist_r_out_ratio": 0.4059,
            "eye_dist_ratio": -0.4321,
            "face_len": 0.3855,
            "forehead_width": 0.1931,
            "zygomatic": 0.1154,
            "mandible_width": 0.0821,
            "mandible_angle": 0.0433,
            "fwhr": -0.1303,
            "golden_angle": -0.4047,
            "eyebrow_h": 0.1128,
            "eyebrow_w": 0.1656,
            "eyebrow_area": 0.2944,
            "eyebrow_thick": -0.1894,
            "eyebrow_tilt": 1.4286,
            "eyebrow_curvature": 1.9053,
            "eye_length": 0.2775,
            "eye_width": 0.7607,
            "eye_area": 0.7741,
            "inner_eye_angle": 0.4720,
            "relative_eye_area": 0.5970,
            "eyelid_tilt": 0.0332,
            "eyebrow_eye_length": -0.3636,
            "nose_ala": -0.2050,
            "philtrum_length": 0.0516,
            "lip_h": -0.1773,
            "lip_w": 0.1091,
            "lip_thick": 0.0662,
            "lip_area": 0.0850,
            "mouth_angle": 0.5327,
            "chin_len": 0.5574,
            "chin_width": 0.0448,
            "chin_angle": -0.3671,
            "face_divergence": -0.2287,
        }
    elif gender == "female":
        beauty_face = {
            "top": -0.0241,
            "mid": -0.1940,
            "down": 0.1362,
            "dist_r_out_ratio": 0.4059,
            "eye_dist_ratio": -0.4321,
            "face_len": 0.3855,
            "forehead_width": 0.1931,
            "zygomatic": 0.1154,
            "mandible_width": 0.0821,
            "mandible_angle": 0.0433,
            "fwhr": -0.1303,
            "golden_angle": -0.4047,
            "eyebrow_h": 0.1128,
            "eyebrow_w": 0.1656,
            "eyebrow_area": 0.2944,
            "eyebrow_thick": -0.1894,
            "eyebrow_tilt": 1.4286,
            "eyebrow_curvature": 1.9053,
            "eye_length": 0.2775,
            "eye_width": 0.7607,
            "eye_area": 0.7741,
            "inner_eye_angle": 0.4720,
            "relative_eye_area": 0.5970,
            "eyelid_tilt": 0.0332,
            "eyebrow_eye_length": -0.3636,
            "nose_ala": -0.2050,
            "philtrum_length": 0.0516,
            "lip_h": -0.1773,
            "lip_w": 0.1091,
            "lip_thick": 0.0662,
            "lip_area": 0.0850,
            "mouth_angle": 0.5327,
            "chin_len": 0.5574,
            "chin_width": 0.0448,
            "chin_angle": -0.3671,
            "face_divergence": -0.2287,
        }
    return beauty_face[beauty_cols_name]


def generate_ideal_face(gender, z_scores):
    """根据Z分数生成理想人脸特征向量"""
    ideal_features = {}
    for feature_name, z_score in z_scores.items():
        # 上庭占比Z分数
        if feature_name not in ["top", "eyebrow_h", "eyebrow_w", "eyebrow_area", "eyebrow_thick", "eyebrow_tilt",
                                "eyebrow_curvature", "eye_length", "eye_width", "eye_area"]:  # 不为 eye、mouth、face 时跳过
            continue
        # ideal_features[feature_name] = z_score * (1/3)
        ideal_features[feature_name] = beauty_face_score(gender, feature_name) + (0 - z_score) * (1 / 3)
    return ideal_features


def calculate_euclidean_distance(features1, features2):
    """计算两个特征向量的欧几里得距离"""
    # 获取共同的特征名
    common_features = set(features1.keys()) & set(features2.keys())

    if not common_features:
        return float('inf')

    # 计算欧几里得距离
    distance_squared = 0
    for feature in common_features:
        diff = features1[feature] - features2[feature]
        distance_squared += diff * diff

    return math.sqrt(distance_squared)


def process_similarity_analysis(ideal_face_features, folder_path, detector, gender):
    """处理文件夹中所有图片与理想人脸的相似度分析"""
    image_paths = []
    if os.path.isdir(folder_path):
        image_paths = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
    elif os.path.isfile(folder_path):
        if folder_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths = [folder_path]

    results = []

    for img_path in image_paths:
        z_scores, status = extract_face_features(img_path, detector, gender)
        if z_scores is not None:
            # 计算与理想人脸的距离
            distance = calculate_euclidean_distance(ideal_face_features, z_scores)
            results.append({
                "image_name": os.path.basename(img_path),
                "image_path": img_path,
                "distance": round(distance, 4),
                "status": status
            })
        else:
            results.append({
                "image_name": os.path.basename(img_path),
                "image_path": img_path,
                "distance": float('inf'),
                "status": status
            })

    # 按距离排序（距离越小越相似）
    results.sort(key=lambda x: x["distance"])

    return results


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("❌ 参数错误！")
        print(
            "使用格式: python IdealFaceGenerator.py <理想人脸图片路径> <待比较文件夹路径> <文件夹性别> <基准人脸性别>")
        print("示例: python IdealFaceGenerator.py ideal.jpg photos/ male female")
        sys.exit(1)

    ideal_face_path = sys.argv[1]
    compare_folder_path = sys.argv[2]
    folder_gender = sys.argv[3].lower()
    ideal_face_gender = sys.argv[4].lower()

    if folder_gender not in ['male', 'female']:
        print(f"❌ 文件夹性别参数错误：'{folder_gender}'，请使用 male 或 female")
        sys.exit(1)

    if ideal_face_gender not in ['male', 'female']:
        print(f"❌ 基准人脸性别参数错误：'{ideal_face_gender}'，请使用 male 或 female")
        sys.exit(1)

    # 模型初始化
    model_path = "scripts/models/face_landmarker.task"
    if not os.path.exists(model_path):
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在：{model_path}")
            sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1, output_face_blendshapes=True)
    detector = vision.FaceLandmarker.create_from_options(options)

    print("🔍 正在生成理想人脸特征...")
    # 1. 提取理想人脸的Z分数（使用基准人脸性别）
    ideal_z_scores, status = extract_face_features(ideal_face_path, detector, ideal_face_gender)

    if ideal_z_scores is None:
        print(f"❌ 理想人脸处理失败: {status}")
        detector.close()
        sys.exit(1)

    print("✅ 理想人脸特征提取完成")

    # 2. 生成理想人脸特征向量
    ideal_face_features = generate_ideal_face(folder_gender, ideal_z_scores)
    print("✅ 理想人脸向量生成完成")

    # 3. 处理文件夹中的所有图片（使用文件夹性别）
    print("🔍 正在分析文件夹中的图片...")
    similarity_results = process_similarity_analysis(ideal_face_features, compare_folder_path, detector, folder_gender)

    print("\n" + "=" * 50)
    print("相似度分析结果（按相似度排序）:")
    print("=" * 50)

    for i, result in enumerate(similarity_results, 1):
        if result["distance"] != float('inf'):
            print(f"{i:2d}. {result['image_name']:<30} 距离: {result['distance']:<8.4f}")
        else:
            print(f"{i:2d}. {result['image_name']:<30} 无法处理: {result['status']}")

    print("=" * 50)
    print(f"总共处理了 {len(similarity_results)} 张图片")

    output_data = {
        "ideal_face_path": ideal_face_path,
        "compare_folder_path": compare_folder_path,
        "folder_gender": folder_gender,
        "ideal_face_gender": ideal_face_gender,
        "ideal_face_features": ideal_face_features,
        "similarity_results": similarity_results
    }

    output_file = "similarity_analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"📊 结果已保存到: {output_file}")

    detector.close()
    print("✅ 处理完成！")