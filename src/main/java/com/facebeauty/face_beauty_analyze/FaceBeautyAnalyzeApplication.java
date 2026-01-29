package com.facebeauty.face_beauty_analyze;

import lombok.extern.slf4j.Slf4j;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.facebeauty.face_beauty_analyze.mapper")
@Slf4j
public class FaceBeautyAnalyzeApplication {

    public static void main(String[] args) {
        SpringApplication.run(FaceBeautyAnalyzeApplication.class, args);
        log.info("项目启动成功...");
    }

}
