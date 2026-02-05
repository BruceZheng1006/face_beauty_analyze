package com.facebeauty.face_beauty_analyze.controller;

import com.facebeauty.face_beauty_analyze.entity.FaceAnalysisResult;
import com.facebeauty.face_beauty_analyze.service.FaceAnalysisResultService;
import com.facebeauty.face_beauty_analyze.utils.PythonFaceAnalyzeUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.nio.file.StandardCopyOption;

@RestController
public class FaceAnalyzeController {

    @Autowired
    private FaceAnalysisResultService faceAnalysisResultService;

    // 临时文件存储目录（用于处理过程中的临时文件）
    private static final String TEMP_UPLOAD_DIR = System.getProperty("user.dir") + File.separator + "temp";

    // 1. 访问根路径，返回前端页面
    @GetMapping("/")
    public ModelAndView index() {
        return new ModelAndView("index");
    }

    // 2. 新增：文件上传并处理接口
    @PostMapping("/analyze/upload")
    public Map<String, Object> analyzeUploadedFile(@RequestParam("file") MultipartFile file, @RequestParam("gender") String gender) {
        Map<String, Object> result = new HashMap<>();
        
        // 1. 校验上传文件是否为空
        if (file.isEmpty()) {
            result.put("success", false);
            result.put("message", "❌ 请选择要上传的图片文件！");
            return result;
        }

        // 2. 初始化临时目录（不存在则创建）
        File tempDir = new File(TEMP_UPLOAD_DIR);
        if (!tempDir.exists()) {
            tempDir.mkdirs();
        }

        // 3. 保存上传的文件到临时目录（生成唯一文件名，避免重复）
        String originalFileName = file.getOriginalFilename();
        String uniqueFileName = UUID.randomUUID().toString() + "_" + originalFileName;
        File tempFile = new File(tempDir, uniqueFileName);
        
        try {
            // 保存上传的文件到本地临时目录
            file.transferTo(tempFile);
        } catch (IOException e) {
            result.put("success", false);
            result.put("message", "❌ 保存上传文件失败：" + e.getMessage());
            return result;
        }

        // 4. 根据性别调用不同的Python脚本处理临时文件
        String pythonResult = PythonFaceAnalyzeUtil.callFaceAnalysisPython(tempFile.getAbsolutePath(), gender);
        
        // 5. 将Python结果解析为JSON格式
        String jsonResult = PythonFaceAnalyzeUtil.parsePythonOutputToJson(pythonResult);

        // 6. 创建人脸分析结果实体并保存到数据库（不包含图像路径）
        FaceAnalysisResult faceAnalysisResult = new FaceAnalysisResult();
        faceAnalysisResult.setImageName(originalFileName); // 保存原始文件名
        
        // 从原始Python输出中提取人脸数量
        int faceCount = extractFaceCountFromOutput(pythonResult);
        faceAnalysisResult.setFaceCount(faceCount);
        
        // 如果没有人脸，也要保存分析结果
        faceAnalysisResult.setAnalysisJson(jsonResult);

        // 7. 保存到数据库
        int saveResult = faceAnalysisResultService.saveFaceAnalysisResult(faceAnalysisResult);
        
        // 8. 删除临时文件（立即删除上传的图片，确保隐私）
        tempFile.delete();

        // 9. 返回处理结果
        if (saveResult > 0) {
            result.put("success", true);
            result.put("message", "分析完成，结果已保存到数据库");
            result.put("data", jsonResult);
            result.put("id", faceAnalysisResult.getId());
        } else {
            result.put("success", false);
            result.put("message", "分析完成但保存到数据库失败");
            result.put("data", jsonResult);
        }
        
        return result;
    }

    // 3. 根据ID查询分析结果
    @GetMapping("/analyze/result/{id}")
    public Map<String, Object> getAnalysisResult(@PathVariable Long id) {
        Map<String, Object> result = new HashMap<>();
        
        FaceAnalysisResult faceAnalysisResult = faceAnalysisResultService.getFaceAnalysisResultById(id);
        if (faceAnalysisResult != null) {
            result.put("success", true);
            result.put("data", faceAnalysisResult);
        } else {
            result.put("success", false);
            result.put("message", "未找到对应的结果");
        }
        
        return result;
    }

    // 提取人脸数量的方法
    private int extractFaceCountFromOutput(String pythonOutput) {
        String[] lines = pythonOutput.split("\n");
        int faceCount = 0;
        
        for (String line : lines) {
            line = line.trim();
            if (line.contains("第") && line.contains("张人脸")) {
                faceCount++;
            }
            // 如果发现没有检测到人脸的信息，直接返回0
            if (line.contains("未检测到人脸") || line.contains("ERR_NO_FACE_DETECTED")) {
                return 0;
            }
        }
        
        return faceCount;
    }

    // 保留原来的路径访问接口（用于本地测试）
    @GetMapping("/analyze")
    public String analyzeFace(@RequestParam("inputPath") String inputPath) {
        return PythonFaceAnalyzeUtil.callFaceAnalysisPython(inputPath);
    }
}