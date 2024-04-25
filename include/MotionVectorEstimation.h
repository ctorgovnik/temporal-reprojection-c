#ifndef MOTIONVECTORESTIMATION_H
#define MOTIONVECTORESTIMATION_H
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>


using MotionVector = std::tuple<std::pair<int, int>, std::pair<int, int>>; // ((x, y), (dx, dy))
using GrayFrame = std::vector<std::vector<float>>;


class MotionVectorEstimation{
private:

    /**
     * compute MAD between two blocks of pixels
     * 
     * @param prevBlock grayscale block from previous frame. blockSize x blockSize 
     * @param currBlock grayscale block from current frame
     * @return the MAD between the two blocks
     * 
    */
    float computeMad(
        const cv::Mat& prevBlock, 
        const cv::Mat& currBlock
    ); 

    /**
     * get a block from a grayscale frame
     * 
     *@param frame grayscale frame
     *@param startRow starting row
     *@param startCol starting column
     *@param blockSize size of blocks
     *@return a blockSize x blockSize matrix from frame
    */
    cv::Mat getBlock(const cv::Mat& frame, int startRow, int startCol, int blockSize);

    std::vector<cv::Mat> buildResolutionPyramid(const cv::Mat& frame, int levels);

    cv::Mat warpFrame(const cv::Mat& frame, const std::vector<std::vector<MotionVector>>& motionVectors, int blockSize);

    void workerFunction(const cv::Mat& prevFrame, const cv::Mat& currFrame, 
                                            int blockSize, int searchRange,
                                            int startRow, int endRow, std::vector<std::vector<MotionVector>>& motionVectors);

public:

    /**
     * estimate motion vectors between two frames
     * 
     * @param prevFrame matrix of previous frame's pixels (grayscale). Height x Width
     * @param currFrame matrix of current frame's pixels (grayscale).  Height x Width
     * @param blockSize size of each block in the frame. Blocks of blockSize x blockSize
     * @param searchRange range to search around block in prevFrame. height: [-searchRange, searchRange], width: [-searchRange, searchRange]
     * @return matrix of motion vectors corresponding to each blocks movement from prevFrame to currFrame. height x width x 2. Each entry contains two tuples: [(y_0, x_0), (dy, dx)]
    */
    std::vector<std::vector<MotionVector>> estimateMotionVectors(
        const cv::Mat& prevFrame, 
        const cv::Mat& currFrame, 
        int blockSize=16, 
        int searchRange=2
    ); 

    std::vector<std::vector<MotionVector>> estimateMotionVectorsPyramid(
        const cv::Mat& prevFrame, 
        const cv::Mat& currFrame, 
        int blockSize, 
        int searchRange, 
        int levels
    );


    std::vector<std::vector<MotionVector>> estimateAndRefineMotionVectors(
    const cv::Mat& prevFrame, 
    const cv::Mat& currFrame, 
    int blockSize, 
    int searchRange, 
    int levels);


    


};

#endif