#ifndef MOTION_VECTOR_ESTIMATION_H
#define MOTION_VECTOR_ESTIMATION_H
#include <vector>
#include <tuple>

using MotionVector = std::tuple<std::pair<int, int>, std::pair<int, int>>; // ((x, y), (dx, dy))

class MotionVectorEstimation{
private:

    /**
     * compute MAD between two blocks of pixels
     * 
     * @param prevBlock block from previous frame. blockSize x blockSize x 3
     * @param currBlock block from current frame
     * @return the MAD between the two blocks
     * 
    */
    float computeMad(
        const std::vector<std::vector<std::vector<float>>>& prevBlock, 
        const std::vector<std::vector<std::vector<float>>>& currBlock
    ); 

public:

    /**
     * estimate motion vectors between two frames
     * 
     * @param prevFrame matrix of previous frame's pixels (grayscale). Height x Width x 3
     * @param currFrame matrix of current frame's pixels (grayscale).  Height x Width x 3
     * @param blockSize size of each block in the frame. Blocks of blockSize x blockSize x 3
     * @param searchRange range to search around block in prevFrame. height: [-searchRange, searchRange], width: [-searchRange, searchRange]
     * @return matrix of motion vectors corresponding to each blocks movement from prevFrame to currFrame. height x width x 2. Each entry contains two tuples: [(y_0, x_0), (dy, dx)]
    */
    std::vector<std::vector<MotionVector>> estimateMotionVectors(
        const std::vector<std::vector<std::vector<float>>>& prevFrame, 
        const std::vector<std::vector<std::vector<float>>>& currFrame, 
        int blockSize=16, 
        int searchRange=2
    ); 


};

#endif