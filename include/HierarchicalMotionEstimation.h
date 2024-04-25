#ifndef HIERARCHICALMOTIONESTIMATION_H
#define HIERARCHICALMOTIONESTIMATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

class HierarchicalMotionEstimation {
public:
    HierarchicalMotionEstimation();
    
    // Main function to estimate motion vectors
    std::vector<std::vector<std::pair<int, int>>> estimateMotionVectors(
        const cv::Mat& prevFrame, 
        const cv::Mat& currFrame, 
        int blockSize, 
        int searchRange);

    cv::Mat reprojectImage(const cv::Mat& image, const std::vector<std::vector<std::pair<int, int>>>& motionVectors);

private:
    // Function to create a pyramid of downscaled images
    std::vector<cv::Mat> buildResolutionPyramid(const cv::Mat& frame, int levels);

    std::vector<std::vector<std::pair<int, int>>> initializeMotionVectorsForLevel(
    const cv::Mat& frame, int blockSize);

    cv::Mat getBlock(const cv::Mat& frame, int startRow, int startCol, int blockSize);

    // Function to warp an image based on motion vectors
    cv::Mat warpImage(const cv::Mat& image, const std::vector<std::vector<std::pair<int, int>>>& motionVectors, int blockSize);
    
    // Compute the distance between patches (e.g., using Mean Absolute Difference)
    float computePatchDistance(const cv::Mat& patch1, const cv::Mat& patch2);
    
    // Function to refine motion vectors at a given pyramid level
    std::vector<std::vector<std::pair<int, int>>> refineMotionVectorsAtLevel(
        const cv::Mat& prevLevel, 
        const cv::Mat& currLevel, 
        const std::vector<std::vector<std::pair<int, int>>>& coarseMotionVectors, 
        int blockSize, 
        int searchRange);
};

#endif // MOTIONESTIMATION_H
