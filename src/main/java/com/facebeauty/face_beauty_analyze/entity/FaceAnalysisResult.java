package com.facebeauty.face_beauty_analyze.entity;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import java.time.LocalDateTime;

@Data
public class FaceAnalysisResult {

    private Long id; // 主键ID

    private String imageName; // 图片原始名称

    private String analysisJson; // 人脸分析结果JSON数据


    private Integer faceCount; // 检测到的人脸数量

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createdTime; // 创建时间

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updatedTime; // 更新时间
}