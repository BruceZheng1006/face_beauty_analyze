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

# 强制标准输出/错误输出使用UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 美的人脸参数
MALE_STATS = {
    # 第一行数据作为 mean，第二行作为 std
    "top": {"mean": 0.184579412, "std": 0.019263556},  # 上庭占比
    "mid": {"mean": 0.416982353, "std": 0.017155371},  # 中庭占比
    "down": {"mean": 0.398447059, "std": 0.029449271},  # 下庭占比
    "dist_r_out_ratio": {"mean": 0.250844118, "std": 0.09688761},  # 右眼外侧留白
    "eye_length": {"mean": 26.93441176, "std": 3.380042804},  # 右眼长度(mm)
    "eye_dist_ratio": {"mean": 0.259317647, "std": 0.023423835},  # 内眼角间距占比
    "zygomatic": {"mean": 137.4279412, "std": 7.664152146},  # 颧骨宽度(mm)
    "face_len": {"mean": 169.6982353, "std": 12.4668012},  # 脸部长度(mm)
    "forehead_width": {"mean": 139.2911765, "std": 7.106419536},  # 颞部宽度(mm)
    "mandible_width": {"mean": 108.4605882, "std": 6.449212419},  # 下颌角宽度(mm)
    "mandible_angle": {"mean": 143.4979412, "std": 6.057050719},  # 下颌角度数(°)
    "fwhr": {"mean": 1.60615, "std": 0.162913155},  # 面部宽高比
    "golden_angle": {"mean": 69.41411765, "std": 10.95451723},  # 黄金三角度数
    "eyebrow_h": {"mean": 11.74647059, "std": 6.242071622},  # 眉毛高度(mm)

    # 第二张图数据
    "eyebrow_w": {"mean": 43.68264706, "std": 9.083454677},  # 眉毛长度(mm)
    "eyebrow_area": {"mean": 541.07, "std": 357.3057984},  # 眉毛大小(mm²)
    "eyebrow_thick": {"mean": 10.77205882, "std": 2.132314097},  # 眉毛粗细(mm)
    "eyebrow_tilt": {"mean": 11.63058824, "std": 10.10328629},  # 眉毛挑度(°)
    "eyebrow_curvature": {"mean": 10.88117647, "std": 11.50156889},  # 眉毛弯度(°)
    "eye_length": {"mean": 26.93441176, "std": 3.380042804},  # 眼睛长度(mm)
    "eye_width": {"mean": 9.537352941, "std": 1.545938338},  # 眼睛宽度(mm)
    "eye_area": {"mean": 257.6061765, "std": 58.87479497},  # 眼睛面积(mm²)
    "inner_eye_angle": {"mean": 50.35147059, "std": 8.101703573},  # 内眦角度数(°)
    "relative_eye_area": {"mean": 0.013182588, "std": 0.002773578},  # 相对眼面积
    "eyelid_tilt": {"mean": 4.375588235, "std": 13.86158449},  # 眼睑裂倾斜度
    "nose_ala": {"mean": 40.78794118, "std": 1.907045419},  # 鼻翼宽度(mm)
    "philtrum_length": {"mean": 15.37823529, "std": 1.748689376},  # 人中长度(mm)

    # 第三张图数据
    "lip_h": {"mean": 20.13323529, "std": 3.719940481},  # 嘴唇高度(mm)
    "lip_w": {"mean": 49.02970588, "std": 5.028517198},  # 嘴唇长度(mm)
    "lip_thick": {"mean": 9.395588235, "std": 1.693517382},  # 嘴唇厚度(mm)
    "lip_area": {"mean": 917.1479412, "std": 164.7666426},  # 嘴巴大小(mm²)
    "mouth_angle": {"mean": 101.5258824, "std": 9.754326376},  # 嘴角弯曲度(°)
    "chin_len": {"mean": 33.20852941, "std": 5.624084831},  # 下巴长度(mm)
    "chin_width": {"mean": 61.41294118, "std": 3.425967042},  # 下巴宽度(mm)
    "chin_angle": {"mean": 137.2311765, "std": 12.73631879},  # 下巴角度(°)
    "face_divergence": {"mean": 0.30575, "std": 0.015282348},  # 面部聚散度
}

FEMALE_STATS = {
    # 三庭
    "top": {"mean": 0.195764, "std": 0.018924553},  # 上庭占比
    "mid": {"mean": 0.439754, "std": 0.023449471},  # 中庭占比
    "down": {"mean": 0.36448, "std": 0.034285927},  # 下庭占比
    # 五眼
    "dist_r_out_ratio": {"mean": 0.206718, "std": 0.09558037},  # 右眼外侧留白占比
    "eye_dist_ratio": {"mean": 0.266722, "std": 0.016134873},  # 内眼角间距占比
    # 脸部
    "face_len": {"mean": 162.7054, "std": 10.5180067},  # 脸部长度(mm)
    "forehead_width": {"mean": 134.484, "std": 6.370166089},  # 颞部宽度(mm)
    "zygomatic": {"mean": 132.5246, "std": 6.378261271},  # 颧骨宽度(mm)
    "mandible_width": {"mean": 102.8512, "std": 6.117873369},  # 下颌角宽度(mm)
    "mandible_angle": {"mean": 146.8932, "std": 6.2539058},  # 下颌角度数(°)
    "fwhr": {"mean": 1.569264, "std": 0.121378942},  # 面部宽高比
    "golden_angle": {"mean": 67.3638, "std": 7.220087781},  # 黄金三角度数
    # 眉毛
    "eyebrow_h": {"mean": 11.8394, "std": 5.360706077},  # 眉毛高度(mm)
    "eyebrow_w": {"mean": 41.7424, "std": 9.937337583},  # 眉毛长度(mm)
    "eyebrow_area": {"mean": 507.3924, "std": 314.1225082},  # 眉毛大小(mm²)
    "eyebrow_thick": {"mean": 10.6362, "std": 1.861317695},  # 眉毛粗细(mm)
    "eyebrow_tilt": {"mean": 11.2592, "std": 9.580187856},  # 眉毛挑度(°)
    "eyebrow_curvature": {"mean": 8.9622, "std": 11.05982229},  # 眉毛弯度(°)
    # 眼睛
    "eye_length": {"mean": 27.8838, "std": 3.331849871},  # 右眼长度(mm)
    "eye_width": {"mean": 11.1502, "std": 1.533801147},  # 眼睛宽度(mm)
    "eye_area": {"mean": 313.659, "std": 71.65914575},  # 眼睛面积(mm²)
    "inner_eye_angle": {"mean": 58.048, "std": 7.657635144},  # 内眦角度数(°)
    "relative_eye_area": {"mean": 0.01754324, "std": 0.003913969},  # 相对眼面积
    "eyelid_tilt": {"mean": 5.6548, "std": 13.98465763},  # 眼睑裂倾斜度(°)
    # 鼻子
    "nose_ala": {"mean": 39.652, "std": 2.323069521},  # 鼻翼宽度(mm)
    "philtrum_length": {"mean": 13.2166, "std": 1.492374765},  # 人中长度(mm)
    # 嘴唇
    "lip_h": {"mean": 20.2864, "std": 5.026603529},  # 嘴唇高度(mm)
    "lip_w": {"mean": 49.2906, "std": 6.143733852},  # 嘴唇长度(mm)
    "lip_thick": {"mean": 9.1392, "std": 1.599738529},  # 嘴唇厚度(mm)
    "lip_area": {"mean": 896.0962, "std": 171.7643857},  # 嘴巴大小(mm²)
    "mouth_angle": {"mean": 106.3872, "std": 13.52027338},  # 嘴角弯曲度(°)
    # 下巴
    "chin_len": {"mean": 26.5866, "std": 4.503211125},  # 下巴长度(mm)
    "chin_width": {"mean": 55.7584, "std": 4.207035232},  # 下巴宽度(mm)
    "chin_angle": {"mean": 133.9166, "std": 6.860110089},  # 下巴角度(°)
    # 面部聚散度
    "face_divergence": {"mean": 0.324128, "std": 0.017739459},  # 面部聚散度
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
        face_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in face_contour_indices], dtype=np.int32)
        face_area = cv2.contourArea(face_pts) * (factor ** 2)
        features["relative_eye_area"] = eye_area / face_area
        # features["inner_eye_angle"] = get_angle(landmarks[157], landmarks[133], landmarks[154], img_w, img_h)
        # features["eyelid_tilt"] = get_line_tilt(landmarks[33], landmarks[133], img_w, img_h)
        
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
                z_scores[feature_name] = calculate_z_score(value, stats[feature_name]["mean"], stats[feature_name]["std"])
            else:
                z_scores[feature_name] = 0.0
        
        return z_scores, "成功"
        
    except Exception as e:
        return None, f"处理失败: {str(e)}"

def generate_ideal_face(z_scores):
    """根据Z分数生成理想人脸特征向量"""
    ideal_features = {}
    for feature_name, z_score in z_scores.items():
        ideal_features[feature_name] = z_score * (1/3)
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
    """
    使用说明：
    命令行调用格式：
    python IdealFaceGenerator.py <理想人脸图片路径> <待比较文件夹路径> <文件夹性别> <基准人脸性别>
    
    示例：
    python IdealFaceGenerator.py "ideal_face.jpg" "test_photos" male female
    """
    
    if len(sys.argv) != 5:
        print("❌ 参数错误！")
        print("使用格式: python IdealFaceGenerator.py <理想人脸图片路径> <待比较文件夹路径> <文件夹性别> <基准人脸性别>")
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
        model_path = "models/face_landmarker.task"  # 备用路径
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
    ideal_face_features = generate_ideal_face(ideal_z_scores)
    print("✅ 理想人脸向量生成完成")
    
    # 3. 处理文件夹中的所有图片（使用文件夹性别）
    print("🔍 正在分析文件夹中的图片...")
    similarity_results = process_similarity_analysis(ideal_face_features, compare_folder_path, detector, folder_gender)
    
    # 4. 输出结果
    print("\n" + "="*50)
    print("相似度分析结果（按相似度排序）:")
    print("="*50)
    
    for i, result in enumerate(similarity_results, 1):
        if result["distance"] != float('inf'):
            print(f"{i:2d}. {result['image_name']:<30} 距离: {result['distance']:<8.4f}")
        else:
            print(f"{i:2d}. {result['image_name']:<30} 无法处理: {result['status']}")
    
    print("="*50)
    print(f"总共处理了 {len(similarity_results)} 张图片")
    
    # 保存结果为JSON格式
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