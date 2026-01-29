-- 更新数据库表结构，移除beauty_score字段并说明image_path字段用途
-- 注意：在实际部署时，请根据您的数据库情况执行以下SQL

-- 方式1：如果表中已经有beauty_score列，则删除它
-- ALTER TABLE face_analysis_results DROP COLUMN beauty_score;

-- 方式2：如果需要创建新表，则使用以下结构
/*
CREATE TABLE face_analysis_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    image_name VARCHAR(255) NOT NULL COMMENT '图片原始名称',
    image_path VARCHAR(500) NOT NULL COMMENT '图片原始文件存储路径',
    analysis_json JSON NOT NULL COMMENT '人脸分析结果JSON数据',
    face_count INT DEFAULT 0 COMMENT '检测到的人脸数量',
    analysis_status TINYINT DEFAULT 1 COMMENT '分析状态：1-成功，0-失败',
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_created_time (created_time),
    INDEX idx_image_name (image_name)
) COMMENT='人脸美学分析结果表';
*/

-- 如果需要增加image_path字段的长度以容纳更长的路径
-- ALTER TABLE face_analysis_results MODIFY COLUMN image_path VARCHAR(1000) NOT NULL COMMENT '图片原始文件存储路径';