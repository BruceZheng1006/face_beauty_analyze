package com.facebeauty.face_beauty_analyze.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Python人脸分析脚本调用工具类
 */
public class PythonFaceAnalyzeUtil {
    // 1. 配置关键路径（保持你原来的路径不变，根据实际情况修改）
    private static final String PYTHON_SCRIPT_PATH = "D:\\Bias\\20.3\\face_beauty_analyze\\scripts\\python\\ZscoreAllwithbanshen.py";
    private static final String ZSCORE_EXCEL_SCRIPT_PATH = "D:\\Bias\\20.3\\face_beauty_analyze\\scripts\\python\\zscoreexcelwithuandzws.py";
    private static final String IDEAL_FACE_SCRIPT_PATH = "D:\\Bias\\20.3\\face_beauty_analyze\\scripts\\python\\IdealFaceGenerator.py";
    private static final String PYTHON_COMMAND = "python";

    /**
     * 调用Python脚本，获取所有打印输出结果
     * @param inputPath 传入的图片路径/文件夹路径
     * @return Python的所有打印输出内容
     */
    public static String callFaceAnalysisPython(String inputPath) {
        return callFaceAnalysisPython(inputPath, "male"); // 默认为男性
    }
    
    /**
     * 根据性别调用Python脚本，获取所有打印输出结果
     * @param inputPath 传入的图片路径/文件夹路径
     * @param gender 性别（"male" 或 "female"）
     * @return Python的所有打印输出内容
     */
    public static String callFaceAnalysisPython(String inputPath, String gender) {
        // 空值校验，避免空指针
        if (inputPath == null || inputPath.trim().isEmpty()) {
            return "❌ 传入的路径不能为空！";
        }
        
        if (gender == null || gender.trim().isEmpty()) {
            gender = "male"; // 默认为男性
        }

        List<String> command = new ArrayList<>();
        command.add(PYTHON_COMMAND);
        command.add(PYTHON_SCRIPT_PATH);
        command.add(inputPath);
        command.add(gender);  // 添加性别参数

        StringBuilder pythonOutput = new StringBuilder();
        Process process = null;
        BufferedReader reader = null;

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            process = processBuilder.start();

            // 指定UTF-8编码，避免中文乱码
            reader = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                pythonOutput.append(line).append(System.lineSeparator());
            }

            // 等待脚本执行完成
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                pythonOutput.append(System.lineSeparator()).append("❌ Python脚本执行异常，退出码：").append(exitCode);
            }

        } catch (IOException e) {
            return "❌ 调用Python脚本时发生IO异常：" + e.getMessage();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "❌ Python脚本执行进程被中断：" + e.getMessage();
        } finally {
            // 关闭资源
            try {
                if (reader != null) {
                    reader.close();
                }
                if (process != null) {
                    process.destroy();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return pythonOutput.toString();
    }
    
    /**
     * 解析Python脚本输出，转换为JSON格式
     * @param pythonOutput Python脚本的输出文本
     * @return JSON格式的分析结果
     */
    public static String parsePythonOutputToJson(String pythonOutput) {
        ObjectMapper mapper = new ObjectMapper();
        
        try {
            String[] lines = pythonOutput.split("\n");
            
            // 创建一个ObjectNode来存储分析结果
            ObjectNode analysisData = mapper.createObjectNode();
            
            // 用于跟踪当前部分
            String currentSection = "";
            int faceCount = 0;
            
            for (String line : lines) {
                line = line.trim();
                
                // 检查是否是新的人脸段落
                if (line.contains("第") && line.contains("张人脸")) {
                    faceCount++;
                    continue;
                }
                
                // 检查是否包含未检测到人脸的错误信息
                if (line.contains("未检测到人脸") || line.contains("ERR_NO_FACE_DETECTED")) {
                    // 返回空的JSON对象表示没有检测到人脸
                    ObjectNode emptyNode = mapper.createObjectNode();
                    return mapper.writeValueAsString(emptyNode);
                }
                
                // 检查是否是新的分析部分
                if (line.startsWith("●")) {
                    currentSection = line.replace("●", "").trim();
                    continue;
                }
                
                // 检查是否是具体的测量结果
                if (line.contains(":") && !line.startsWith("=") && !line.startsWith("❌") && !line.startsWith("✅️")) {
                    String[] parts = line.split(":", 2);
                    if (parts.length == 2) {
                        String key = parts[0].trim();
                        String value = parts[1].trim();
                        
                        // 移除单位(mm, °等)
                        //value = value.replaceAll("[a-zA-Z°%]+", "").trim();
                        
                        // 添加到对应的部分
                        if (!currentSection.isEmpty()) {
                            ObjectNode sectionNode = (ObjectNode) analysisData.get(currentSection);
                            if (sectionNode == null) {
                                sectionNode = mapper.createObjectNode();
                                analysisData.set(currentSection, sectionNode);
                            }
                            
                            // 尝试将值转换为数字
                            try {
                                if (value.contains(".")) {
                                    sectionNode.put(key, Double.parseDouble(value));
                                } else {
                                    sectionNode.put(key, Integer.parseInt(value));
                                }
                            } catch (NumberFormatException e) {
                                sectionNode.put(key, value); // 如果不是数字就保持原字符串
                            }
                        }
                    }
                }
            }
            
            // 设置解析的数据 - 只返回分析数据部分
            return mapper.writeValueAsString(analysisData);
            
        } catch (Exception e) {
            e.printStackTrace();
            // 如果解析失败，返回错误信息
            ObjectNode errorNode = mapper.createObjectNode();
            errorNode.put("error", "解析Python输出失败: " + e.getMessage());
            try {
                return mapper.writeValueAsString(errorNode);
            } catch (Exception ex) {
                ex.printStackTrace();
                return "{}"; // 返回空JSON对象
            }
        }
    }
    
    /**
     * 调用zscoreexcel.py脚本处理文件夹，生成Excel文件
     * @param folderPath 文件夹路径
     * @param gender 性别（"male" 或 "female"）
     * @param outputPath Excel输出路径
     * @return Python的所有打印输出内容
     */
    public static String callZScoreExcelPython(String folderPath, String gender, String outputPath) {
        // 空值校验，避免空指针
        if (folderPath == null || folderPath.trim().isEmpty()) {
            return "❌ 传入的文件夹路径不能为空！";
        }
        
        if (gender == null || gender.trim().isEmpty()) {
            gender = "male"; // 默认为男性
        }
        
        if (outputPath == null || outputPath.trim().isEmpty()) {
            outputPath = "face_analysis_results.xlsx"; // 默认输出文件名
        }

        List<String> command = new ArrayList<>();
        command.add(PYTHON_COMMAND);
        command.add(ZSCORE_EXCEL_SCRIPT_PATH);
        command.add(folderPath);
        command.add(gender);
        command.add(outputPath);
        
        // 打印执行命令用于调试
        System.out.println("🔧 执行Python命令: " + String.join(" ", command));

        StringBuilder pythonOutput = new StringBuilder();
        Process process = null;
        BufferedReader reader = null;

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            process = processBuilder.start();

            // 指定UTF-8编码，避免中文乱码
            reader = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                pythonOutput.append(line).append(System.lineSeparator());
            }

            // 等待脚本执行完成
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                pythonOutput.append(System.lineSeparator()).append("❌ Python脚本执行异常，退出码：").append(exitCode);
            }

        } catch (IOException e) {
            return "❌ 调用Python脚本时发生IO异常：" + e.getMessage();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "❌ Python脚本执行进程被中断：" + e.getMessage();
        } finally {
            // 关闭资源
            try {
                if (reader != null) {
                    reader.close();
                }
                if (process != null) {
                    process.destroy();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return pythonOutput.toString();
    }
    
    /**
     * 调用理想人脸生成和相似度分析脚本
     * @param idealFacePath 基准人脸图片路径
     * @param compareFolderPath 待比较的文件夹路径
     * @param folderGender 文件夹中人脸的性别（"male" 或 "female"）
     * @param idealFaceGender 基准人脸的性别（"male" 或 "female"）
     * @return Python的所有打印输出内容
     */
    public static String callIdealFacePython(String idealFacePath, String compareFolderPath, String folderGender, String idealFaceGender) {
        // 空值校验，避免空指针
        if (idealFacePath == null || idealFacePath.trim().isEmpty()) {
            return "❌ 传入的基准人脸路径不能为空！";
        }
        
        if (compareFolderPath == null || compareFolderPath.trim().isEmpty()) {
            return "❌ 传入的比较文件夹路径不能为空！";
        }
        
        if (folderGender == null || folderGender.trim().isEmpty()) {
            folderGender = "male"; // 默认为男性
        }
        
        if (idealFaceGender == null || idealFaceGender.trim().isEmpty()) {
            idealFaceGender = "female"; // 默认为女性
        }
        
        if (!folderGender.equals("male") && !folderGender.equals("female")) {
            return "❌ 文件夹性别参数错误：'" + folderGender + "'，请使用 male 或 female";
        }
        
        if (!idealFaceGender.equals("male") && !idealFaceGender.equals("female")) {
            return "❌ 基准人脸性别参数错误：'" + idealFaceGender + "'，请使用 male 或 female";
        }

        List<String> command = new ArrayList<>();
        command.add(PYTHON_COMMAND);
        command.add(IDEAL_FACE_SCRIPT_PATH);
        command.add(idealFacePath);
        command.add(compareFolderPath);
        command.add(folderGender);
        command.add(idealFaceGender);
        
        // 打印执行命令用于调试
        System.out.println("🔧 执行Python命令: " + String.join(" ", command));

        StringBuilder pythonOutput = new StringBuilder();
        Process process = null;
        BufferedReader reader = null;

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            process = processBuilder.start();

            // 指定UTF-8编码，避免中文乱码
            reader = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                pythonOutput.append(line).append(System.lineSeparator());
            }

            // 等待脚本执行完成
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                pythonOutput.append(System.lineSeparator()).append("❌ Python脚本执行异常，退出码：").append(exitCode);
            }

        } catch (IOException e) {
            return "❌ 调用Python脚本时发生IO异常：" + e.getMessage();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "❌ Python脚本执行进程被中断：" + e.getMessage();
        } finally {
            // 关闭资源
            try {
                if (reader != null) {
                    reader.close();
                }
                if (process != null) {
                    process.destroy();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return pythonOutput.toString();
    }
    
    /**
     * 解析理想人脸分析的Python输出，提取JSON结果
     * @param pythonOutput Python脚本的输出文本
     * @return JSON格式的分析结果
     */
    public static String parseIdealFaceOutputToJson(String pythonOutput) {
        ObjectMapper mapper = new ObjectMapper();
        
        try {
            // 查找JSON结果文件的路径
            String[] lines = pythonOutput.split("\n");
            String jsonFilePath = null;
            
            for (String line : lines) {
                line = line.trim();
                if (line.contains("结果已保存到:")) {
                    // 提取文件路径
                    int startIndex = line.indexOf(":") + 1;
                    jsonFilePath = line.substring(startIndex).trim();
                    break;
                }
            }
            
            // 如果找到了JSON文件路径，读取文件内容
            if (jsonFilePath != null && new java.io.File(jsonFilePath).exists()) {
                String jsonContent = new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(jsonFilePath)), "UTF-8");
                return jsonContent;
            } else {
                // 如果没有找到文件，返回原始输出的简化版本
                ObjectNode resultNode = mapper.createObjectNode();
                resultNode.put("raw_output", pythonOutput);
                resultNode.put("status", "completed");
                return mapper.writeValueAsString(resultNode);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
            // 如果解析失败，返回错误信息
            ObjectNode errorNode = mapper.createObjectNode();
            errorNode.put("error", "解析Python输出失败: " + e.getMessage());
            errorNode.put("raw_output", pythonOutput);
            try {
                return mapper.writeValueAsString(errorNode);
            } catch (Exception ex) {
                ex.printStackTrace();
                return "{}"; // 返回空JSON对象
            }
        }
    }
}