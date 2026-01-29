package com.facebeauty.face_beauty_analyze.service.impl;

import com.facebeauty.face_beauty_analyze.entity.FaceAnalysisResult;
import com.facebeauty.face_beauty_analyze.mapper.FaceAnalysisResultMapper;
import com.facebeauty.face_beauty_analyze.service.FaceAnalysisResultService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class FaceAnalysisResultServiceImpl implements FaceAnalysisResultService {

    @Autowired
    private FaceAnalysisResultMapper faceAnalysisResultMapper;

    @Override
    public int saveFaceAnalysisResult(FaceAnalysisResult faceAnalysisResult) {
        faceAnalysisResult.setCreatedTime(LocalDateTime.now());
        faceAnalysisResult.setUpdatedTime(LocalDateTime.now());
        return faceAnalysisResultMapper.insert(faceAnalysisResult);
    }

    @Override
    public FaceAnalysisResult getFaceAnalysisResultById(Long id) {
        return faceAnalysisResultMapper.selectById(id);
    }

    @Override
    public List<FaceAnalysisResult> getAllFaceAnalysisResults() {
        return faceAnalysisResultMapper.selectAll();
    }

    @Override
    public List<FaceAnalysisResult> getFaceAnalysisResultsByImageName(String imageName) {
        return faceAnalysisResultMapper.selectByImageName(imageName);
    }

    @Override
    public int updateFaceAnalysisResult(FaceAnalysisResult faceAnalysisResult) {
        faceAnalysisResult.setUpdatedTime(LocalDateTime.now());
        return faceAnalysisResultMapper.update(faceAnalysisResult);
    }

    @Override
    public int deleteFaceAnalysisResultById(Long id) {
        return faceAnalysisResultMapper.deleteById(id);
    }
}