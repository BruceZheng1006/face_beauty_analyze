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
    
    // 原始文件存储目录（保存上传的原始文件）
    private static final String ORIGINAL_UPLOAD_DIR = System.getProperty("user.dir") + File.separator + "uploads";

    // 1. 访问根路径，返回前端页面
    @GetMapping("/")
    public ModelAndView index() {
        return new ModelAndView("index");
    }

    // 2. 新增：文件上传并处理接口
    @PostMapping("/analyze/upload")
    public Map<String, Object> analyzeUploadedFile(@RequestParam("file") MultipartFile file) {
        Map<String, Object> result = new HashMap<>();
        
        // 1. 校验上传文件是否为空
        if (file.isEmpty()) {
            result.put("success", false);
            result.put("message", "❌ 请选择要上传的图片文件！");
            return result;
        }

        // 2. 初始化临时目录和原始文件目录（不存在则创建）
        File tempDir = new File(TEMP_UPLOAD_DIR);
        if (!tempDir.exists()) {
            tempDir.mkdirs();
        }
        
        File originalDir = new File(ORIGINAL_UPLOAD_DIR);
        if (!originalDir.exists()) {
            originalDir.mkdirs();
        }

        // 3. 保存上传的文件到临时目录（生成唯一文件名，避免重复）
        String originalFileName = file.getOriginalFilename();
        String uniqueFileName = UUID.randomUUID().toString() + "_" + originalFileName;
        File tempFile = new File(tempDir, uniqueFileName);
        
        // 同时保存原始文件到原始文件目录，使用唯一文件名防止覆盖
        String uniqueOriginalFileName = generateUniqueFileName(originalFileName);
        File originalFile = new File(originalDir, uniqueOriginalFileName);
        
        try {
            // 保存上传的文件到本地临时目录
            file.transferTo(tempFile);
            
            // 确保原始文件目录存在
            if (!originalFile.getParentFile().exists()) {
                originalFile.getParentFile().mkdirs();
            }
            
            // 同时保存原始文件（覆盖同名文件）
            // 重要：由于Multipart文件只能transfer一次，我们需要分别处理
            // 先复制临时文件到原始文件位置
            java.nio.file.Files.copy(tempFile.toPath(), originalFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            result.put("success", false);
            result.put("message", "❌ 保存上传文件失败：" + e.getMessage());
            return result;
        }

        // 4. 调用Python脚本处理临时文件（传入临时文件的绝对路径）
        String pythonResult = PythonFaceAnalyzeUtil.callFaceAnalysisPython(tempFile.getAbsolutePath());
        
        // 5. 将Python结果解析为JSON格式
        String jsonResult = PythonFaceAnalyzeUtil.parsePythonOutputToJson(pythonResult);

        // 6. 创建人脸分析结果实体并保存到数据库
        FaceAnalysisResult faceAnalysisResult = new FaceAnalysisResult();
        faceAnalysisResult.setImageName(originalFileName); // 保存原始文件名
        faceAnalysisResult.setImagePath(originalFile.getAbsolutePath()); // 保存唯一命名的原始文件的完整路径
        faceAnalysisResult.setAnalysisJson(jsonResult);
        
        // 从原始Python输出中提取人脸数量
        int faceCount = extractFaceCountFromOutput(pythonResult);
        faceAnalysisResult.setFaceCount(faceCount);

        // 7. 保存到数据库
        int saveResult = faceAnalysisResultService.saveFaceAnalysisResult(faceAnalysisResult);
        
        // 8. （可选）删除临时文件（避免占用磁盘空间）
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

    // 生成唯一文件名的方法
    private String generateUniqueFileName(String originalFileName) {
        String extension = "";
        int lastDotIndex = originalFileName.lastIndexOf('.');
        if (lastDotIndex > 0) {
            extension = originalFileName.substring(lastDotIndex);
        }
        String fileNameWithoutExtension = lastDotIndex > 0 ? 
            originalFileName.substring(0, lastDotIndex) : originalFileName;
        
        return UUID.randomUUID().toString() + "_" + fileNameWithoutExtension + extension;
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
        }
        
        return faceCount;
    }

    // 保留原来的路径访问接口（用于本地测试）
    @GetMapping("/analyze")
    public String analyzeFace(@RequestParam("inputPath") String inputPath) {
        return PythonFaceAnalyzeUtil.callFaceAnalysisPython(inputPath);
    }
}