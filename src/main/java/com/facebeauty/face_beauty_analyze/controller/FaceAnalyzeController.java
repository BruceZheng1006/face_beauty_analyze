package com.facebeauty.face_beauty_analyze.controller;

import com.facebeauty.face_beauty_analyze.entity.FaceAnalysisResult;
import com.facebeauty.face_beauty_analyze.service.FaceAnalysisResultService;
import com.facebeauty.face_beauty_analyze.utils.PythonFaceAnalyzeUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import jakarta.servlet.http.HttpServletResponse;
import java.io.FileInputStream;
import java.io.OutputStream;

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

    // 4. 新增：文件夹上传并生成Excel接口
    @PostMapping("/analyze/folder")
    public void analyzeFolderAndDownloadExcel(
            @RequestParam("files") MultipartFile[] files,
            @RequestParam("gender") String gender,
            HttpServletResponse response) {
        
        File tempFolder = null;
        File excelFile = null;
        
        try {
            // 1. 校验上传文件是否为空
            if (files == null || files.length == 0) {
                response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                response.getWriter().write("{\"success\": false, \"message\": \"❌ 请选择要上传的图片文件！\"}");
                return;
            }

            // 2. 初始化临时目录（不存在则创建）
            File tempDir = new File(TEMP_UPLOAD_DIR);
            if (!tempDir.exists()) {
                tempDir.mkdirs();
            }

            // 3. 创建临时文件夹来存放上传的文件
            String folderName = "temp_folder_" + UUID.randomUUID().toString();
            tempFolder = new File(tempDir, folderName);
            tempFolder.mkdirs();

            // 4. 保存所有上传的文件到临时文件夹（处理相对路径）
            for (MultipartFile file : files) {
                if (!file.isEmpty()) {
                    String originalFileName = file.getOriginalFilename();
                    
                    // 处理相对路径：只取文件名，忽略文件夹路径
                    String fileName = originalFileName;
                    if (originalFileName.contains("\\") || originalFileName.contains("/")) {
                        // 如果包含路径分隔符，只取最后一部分作为文件名
                        String[] pathParts = originalFileName.split("[\\/]+");
                        fileName = pathParts[pathParts.length - 1];
                    }
                    
                    File tempFile = new File(tempFolder, fileName);
                    file.transferTo(tempFile);
                }
            }

            // 5. 调用zscoreexcel.py脚本处理文件夹
            String excelFileName = "face_analysis_results_" + System.currentTimeMillis() + ".xlsx";
            String excelFilePath = tempDir + File.separator + excelFileName;
            
            System.out.println("📁 临时文件夹路径: " + tempFolder.getAbsolutePath());
            System.out.println("👤 性别参数: " + gender);
            System.out.println("📊 Excel输出路径: " + excelFilePath);
            
            String pythonResult = PythonFaceAnalyzeUtil.callZScoreExcelPython(
                tempFolder.getAbsolutePath(), 
                gender, 
                excelFilePath
            );

            // 6. 检查Excel文件是否生成成功
            excelFile = new File(excelFilePath);
            if (!excelFile.exists() || excelFile.length() == 0) {
                response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
                response.getWriter().write("{\"success\": false, \"message\": \"❌ Excel文件生成失败！Python输出：" + pythonResult + "\"}");
                return;
            }

            // 7. 设置响应头，准备下载
            response.setContentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
            response.setHeader("Content-Disposition", "attachment; filename=\"face_analysis_results.xlsx\"");
            response.setContentLength((int) excelFile.length());

            // 8. 传输Excel文件给客户端
            FileInputStream fis = new FileInputStream(excelFile);
            OutputStream os = response.getOutputStream();
            
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = fis.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            
            fis.close();
            os.flush();
            os.close();

        } catch (Exception e) {
            try {
                response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
                response.getWriter().write("{\"success\": false, \"message\": \"❌ 处理过程中发生错误：" + e.getMessage() + "\"}");
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        } finally {
            // 确保无论如何都要清理临时文件和文件夹
            cleanupTempFiles(excelFile, tempFolder);
        }
    }

    // 辅助方法：递归删除文件夹
    private void deleteDirectory(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            directory.delete();
        }
    }
    
    // 辅助方法：清理临时文件
    private void cleanupTempFiles(File excelFile, File tempFolder) {
        try {
            // 删除Excel文件
            if (excelFile != null && excelFile.exists()) {
                boolean deleted = excelFile.delete();
                System.out.println("🗑️ Excel文件清理状态: " + (deleted ? "成功" : "失败"));
            }
            
            // 删除临时文件夹及其内容
            if (tempFolder != null && tempFolder.exists()) {
                deleteDirectory(tempFolder);
                System.out.println("🗑️ 临时文件夹清理完成: " + tempFolder.getAbsolutePath());
            }
        } catch (Exception e) {
            System.err.println("❌ 清理临时文件时发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // 5. 新增：理想人脸生成和相似度分析接口
    @PostMapping("/analyze/ideal-face")
    public Map<String, Object> analyzeIdealFace(
            @RequestParam("idealFile") MultipartFile idealFile,
            @RequestParam("compareFiles") MultipartFile[] compareFiles,
            @RequestParam("folderGender") String folderGender,
            @RequestParam("idealFaceGender") String idealFaceGender) {
        
        Map<String, Object> result = new HashMap<>();
        
        // 1. 校验上传文件是否为空
        if (idealFile.isEmpty()) {
            result.put("success", false);
            result.put("message", "❌ 请选择理想人脸图片！");
            return result;
        }
        
        if (compareFiles == null || compareFiles.length == 0) {
            result.put("success", false);
            result.put("message", "❌ 请选择要比较的图片文件夹！");
            return result;
        }
        
        File tempDir = new File(TEMP_UPLOAD_DIR);
        File idealTempFile = null;
        File compareTempFolder = null;
        
        try {
            // 2. 初始化临时目录
            if (!tempDir.exists()) {
                tempDir.mkdirs();
            }
            
            // 3. 保存理想人脸图片
            String idealFileName = UUID.randomUUID().toString() + "_" + idealFile.getOriginalFilename();
            idealTempFile = new File(tempDir, idealFileName);
            idealFile.transferTo(idealTempFile);
            
            // 4. 创建临时文件夹保存比较图片
            String compareFolderName = "compare_folder_" + UUID.randomUUID().toString();
            compareTempFolder = new File(tempDir, compareFolderName);
            compareTempFolder.mkdirs();
            
            // 5. 保存所有比较图片
            for (MultipartFile file : compareFiles) {
                if (!file.isEmpty()) {
                    String originalFileName = file.getOriginalFilename();
                    String fileName = originalFileName;
                    if (originalFileName.contains("\\") || originalFileName.contains("/")) {
                        String[] pathParts = originalFileName.split("[\\/]+");
                        fileName = pathParts[pathParts.length - 1];
                    }
                    File tempFile = new File(compareTempFolder, fileName);
                    file.transferTo(tempFile);
                }
            }
            
            // 6. 调用Python脚本进行分析
            System.out.println("🔍 理想人脸路径: " + idealTempFile.getAbsolutePath());
            System.out.println("🔍 比较文件夹路径: " + compareTempFolder.getAbsolutePath());
            System.out.println("👤 文件夹性别: " + folderGender + ", 基准人脸性别: " + idealFaceGender);
            
            String pythonResult = PythonFaceAnalyzeUtil.callIdealFacePython(
                idealTempFile.getAbsolutePath(),
                compareTempFolder.getAbsolutePath(),
                folderGender,
                idealFaceGender
            );
            
            // 7. 解析结果
            String jsonResult = PythonFaceAnalyzeUtil.parseIdealFaceOutputToJson(pythonResult);
            
            // 8. 返回结果
            result.put("success", true);
            result.put("message", "分析完成");
            result.put("data", jsonResult);
            result.put("raw_output", pythonResult);
            
        } catch (Exception e) {
            result.put("success", false);
            result.put("message", "❌ 处理过程中发生错误：" + e.getMessage());
            e.printStackTrace();
        } finally {
            // 9. 清理临时文件
            cleanupTempFiles(idealTempFile, compareTempFolder);
        }
        
        return result;
    }
    
    // 保留原来的路径访问接口（用于本地测试）
    @GetMapping("/analyze")
    public String analyzeFace(@RequestParam("inputPath") String inputPath) {
        return PythonFaceAnalyzeUtil.callFaceAnalysisPython(inputPath);
    }
}