package com.facebeauty.face_beauty_analyze.mapper;

import com.facebeauty.face_beauty_analyze.entity.FaceAnalysisResult;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface FaceAnalysisResultMapper {

    /**
     * 插入人脸分析结果
     */
    int insert(FaceAnalysisResult faceAnalysisResult);

    /**
     * 根据ID查询人脸分析结果
     */
    FaceAnalysisResult selectById(Long id);

    /**
     * 查询所有人脸分析结果
     */
    List<FaceAnalysisResult> selectAll();

    /**
     * 根据图片名称查询分析结果
     */
    List<FaceAnalysisResult> selectByImageName(@Param("imageName") String imageName);

    /**
     * 更新人脸分析结果
     */
    int update(FaceAnalysisResult faceAnalysisResult);

    /**
     * 删除人脸分析结果
     */
    int deleteById(Long id);
}