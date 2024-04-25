#include "../include/MotionVectorEstimation.h"
#include <cmath>
#include <limits>
#include <utility>

using namespace std;

using MotionVector = std::tuple<std::pair<int, int>, std::pair<int, int>>;
using GrayFrame = std::vector<std::vector<float>>;


float MotionVectorEstimation::computeMad(const cv::Mat& prevBlock, const cv::Mat& currBlock)
    {
       float mad = 0.0;

       cv::Mat diff;

       cv::absdiff(prevBlock, currBlock, diff);

       cv::Scalar sum = cv::sum(diff);

       return sum[0] / prevBlock.rows * prevBlock.cols;


    }

vector<cv::Mat> MotionVectorEstimation::buildResolutionPyramid(const cv::Mat& frame, int levels) {
    vector<cv::Mat> pyramid;
    pyramid.push_back(frame); // Original frame is the first level of the pyramid

    for (int i = 1; i < levels; ++i) {
        cv::Mat reduced;
        cv::pyrDown(pyramid.back(), reduced); // Downsample the last image in the pyramid
        pyramid.push_back(reduced);
    }

    return pyramid;
}


cv::Mat MotionVectorEstimation::getBlock(const cv::Mat& frame, int startRow, int startCol, int blockSize){
    int effectiveBlockSizeWidth = std::min(blockSize, frame.cols - startCol);
    int effectiveBlockSizeHeight = std::min(blockSize, frame.rows - startRow);
    // std::cout << "StartRow: " << startRow << ", StartCol: " << startCol << ", BlockSize: " << blockSize << std::endl;
    // std::cout << "Frame dimensions: " << frame.rows << "x" << frame.cols << std::endl;

    cv::Rect roi(startCol, startRow, effectiveBlockSizeWidth, effectiveBlockSizeHeight);
    return frame(roi);
}



std::vector<std::vector<MotionVector>> MotionVectorEstimation::estimateMotionVectors( 
    const cv::Mat& prevFrame, 
    const cv::Mat& currFrame, 
    int blockSize, 
    int searchRange){

        int rows = prevFrame.rows;
        int cols = prevFrame.cols;

        // cout << prevFrame.size() << endl;
        // for (int i = 0; i < rows; i++){
        //     for ( int j = 0; i < cols; j++){
        //         cout << "arbitrary channel values of prevFrame" << endl;
        //         cout << prevFrame.at<cv::Vec3f>(i, j)[0] << ", " << prevFrame.at<cv::Vec3f>(i, j)[1] <<", " << prevFrame.at<cv::Vec3f>(i, j)[2]<<endl;
        //     }
        // }
        

        int vectorRows = std::ceil(static_cast<float>(rows) / blockSize);
        int vectorCols = std::ceil(static_cast<float>(cols) / blockSize);
        std::vector<std::vector<MotionVector>> motionVectors(vectorRows, std::vector<MotionVector>(vectorCols, {{0, 0}, {0, 0}}));


        for (int row = 0; row < rows; row += blockSize){
            for (int col = 0; col < cols; col += blockSize){
                cv::Mat currBlock = getBlock(currFrame, row, col, blockSize);

                float bestMad = std::numeric_limits<float>::infinity(); // Set best_mad to infinity
                std::pair<int, int> bestVector = {0, 0}; // Initialize best_vector with (0, 0)
                int reach = 0;

                for (int y = -searchRange; y <= searchRange; ++y){
                    for (int x = -searchRange; x < searchRange; ++x){
                        if (row + y < 0 || col + x < 0 || row + y + blockSize > prevFrame.rows || col + y + blockSize> prevFrame.cols){
                            continue;
                        }
                        cv::Mat prevBlock = getBlock(prevFrame, row + y, col + x, blockSize);
                        if (prevBlock.rows != currBlock.rows || prevBlock.cols != currBlock.cols){
                            continue;
                        }
                        // cout << "block 1 size: " << prevBlock.size() << endl << "Block 2 size: " << currBlock.size() << endl;
                        float mad = computeMad(prevBlock, currBlock);

                        if (mad < bestMad){
                            bestMad = mad;
                            bestVector = {y, x};

                            reach = sqrt((pow(y, 2) + pow(x, 2)));
                        }
                        if (mad == bestMad && reach >  sqrt((pow(y, 2) + pow(x, 2)))){
                            bestVector = {y, x};
                        }
                    }
                }


                 // Assuming row/blockSize and col/blockSize give block indices
                motionVectors[row / blockSize][col / blockSize] = {{row, col}, bestVector};


            }
        }

        return motionVectors;
    }


vector<vector<MotionVector>> MotionVectorEstimation::estimateMotionVectorsPyramid(
    const cv::Mat& prevFrame, 
    const cv::Mat& currFrame, 
    int blockSize, 
    int searchRange, 
    int levels) {

        auto prevPyramid = buildResolutionPyramid(prevFrame, 4);
        auto currPyramid = buildResolutionPyramid(currFrame, 4);

        vector<vector<vector<MotionVector>>> pyramidMotionVectors(levels);

        for (int level = levels - 1; level >= 0; --level) {
            int adjustedBlockSize = blockSize / pow(2, level); // Adjust block size for current level
            int adjustedSearchRange = searchRange; // Search range can be adjusted if needed

            // Use your existing method or adapt it to work with the adjusted block size and search range
            pyramidMotionVectors[level] = estimateMotionVectors(
                prevPyramid[level], currPyramid[level], adjustedBlockSize, adjustedSearchRange);

        // Optionally, implement warping and refinement of motion vectors for this level based on the vectors from coarser levels
        // This step is crucial for leveraging the hierarchical structure
        }

        return pyramidMotionVectors[0];

    }


cv::Mat MotionVectorEstimation::warpFrame(const cv::Mat& frame, const vector<vector<MotionVector>>& motionVectors, int blockSize) {
    cv::Mat warpedFrame = cv::Mat::zeros(frame.size(), frame.type());

    for (int i = 0; i < motionVectors.size(); i++) {
        for (int j = 0; j < motionVectors[i].size(); j++) {
            auto [blockStart, motionVector] = motionVectors[i][j];
            auto [startRow, startCol] = blockStart;
            auto [dy, dx] = motionVector;

            cv::Rect sourceRect(startCol, startRow, blockSize, blockSize);
            cv::Rect targetRect(startCol + dx, startRow + dy, blockSize, blockSize);

            if (targetRect.x >= 0 && targetRect.y >= 0 && targetRect.x + targetRect.width <= frame.cols && targetRect.y + targetRect.height <= frame.rows) {
                frame(sourceRect).copyTo(warpedFrame(targetRect));
            }
        }
    }

    return warpedFrame;
}


vector<vector<MotionVector>> MotionVectorEstimation::estimateAndRefineMotionVectors(
    const cv::Mat& prevFrame, 
    const cv::Mat& currFrame, 
    int blockSize, 
    int searchRange, 
    int levels) {

    auto prevPyramid = buildResolutionPyramid(prevFrame, levels);
    auto currPyramid = buildResolutionPyramid(currFrame, levels);
    vector<vector<MotionVector>> refinedVectors; // Final motion vectors after refinement

    for (int level = levels - 1; level >= 0; --level) {
        // Estimate motion vectors at the current level
        auto motionVectors = estimateMotionVectors(
            prevPyramid[level], currPyramid[level], blockSize >> level, searchRange);

        if (level > 0) {
            // Warp the next finer level of the previous frame based on these vectors
            cv::Mat warpedPrev = warpFrame(prevPyramid[level - 1], motionVectors, blockSize >> (level - 1));
            
            // Refine the motion estimation using the warped previous frame and the next finer level of the current frame
            refinedVectors = estimateMotionVectors(
                warpedPrev, currPyramid[level - 1], blockSize >> (level - 1), searchRange);
        } else {
            refinedVectors = motionVectors; // No refinement needed at the finest level
        }
    }

    return refinedVectors; // Motion vectors for the original resolution
}

