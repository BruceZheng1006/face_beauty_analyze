package com.facebeauty.face_beauty_analyze.service;

import com.facebeauty.face_beauty_analyze.entity.FaceAnalysisResult;

import java.util.List;

public interface FaceAnalysisResultService {

    /**
     * 保存人脸分析结果
     */
    int saveFaceAnalysisResult(FaceAnalysisResult faceAnalysisResult);

    /**
     * 根据ID查询人脸分析结果
     */
    FaceAnalysisResult getFaceAnalysisResultById(Long id);

    /**
     * 查询所有人脸分析结果
     */
    List<FaceAnalysisResult> getAllFaceAnalysisResults();

    /**
     * 根据图片名称查询分析结果
     */
    List<FaceAnalysisResult> getFaceAnalysisResultsByImageName(String imageName);

    /**
     * 更新人脸分析结果
     */
    int updateFaceAnalysisResult(FaceAnalysisResult faceAnalysisResult);

    /**
     * 删除人脸分析结果
     */
    int deleteFaceAnalysisResultById(Long id);
}