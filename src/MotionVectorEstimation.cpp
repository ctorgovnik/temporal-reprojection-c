#include "MotionVectorEstimation.h"
#include <cmath>
#include <limits>
#include <utility>

using namespace std;

using MotionVector = std::tuple<std::pair<int, int>, std::pair<int, int>>;
using GrayFrame = std::vector<std::vector<float>>;


float MotionVectorEstimation::computeMad(const GrayFrame& prevBlock, const GrayFrame& currBlock)
    {
        float mad = 0.0;

        int blockSizeY = prevBlock.size();
        int blockSizeX = prevBlock[0].size();

        for (int y=0; y < blockSizeY; ++y){
            for (int x = 0; x < blockSizeX; ++x){
                mad += abs(currBlock[y][x] - prevBlock[y][x]);
            }
        }

        mad /= (blockSizeX * blockSizeY);

        return mad;


    }

GrayFrame MotionVectorEstimation::getBlock(const GrayFrame& frame, int startRow, int startCol, int blockSize){
    GrayFrame block(blockSize, std::vector<float>(blockSize, 0));

    for (int row = 0; row < blockSize; ++row){
        for (int col = 0; col < blockSize; ++col){
             // Check boundaries to avoid accessing out of range
            if ((startRow + row) < frame.size() && (startCol + col) < frame[0].size()) {
                block[row][col] = frame[startRow + row][startCol + col];
            }
        }
    }

    return block;
}

std::vector<std::vector<MotionVector>> MotionVectorEstimation::estimateMotionVectors( 
    const GrayFrame& prevFrame, 
    const GrayFrame& currFrame, 
    int blockSize=16, 
    int searchRange=2){
        int rows = currFrame.size();
        int cols = currFrame[0].size();

        std::vector<std::vector<MotionVector>> motionVectors(rows / blockSize, std::vector<MotionVector>(cols / blockSize, {{0, 0}, {0, 0}}));


        for (int row = 0; row < rows; row += blockSize){
            for (int col = 0; col < cols; col += blockSize){
                GrayFrame currBlock = getBlock(currFrame, row, col, blockSize);

                float bestMad = std::numeric_limits<float>::infinity(); // Set best_mad to infinity
                std::pair<int, int> bestVector = {0, 0}; // Initialize best_vector with (0, 0)
                int reach = 0;

                for (int y = -searchRange; y <= searchRange; ++y){
                    for (int x = -searchRange; x < searchRange; ++x){
                        GrayFrame prevBlock = getBlock(prevFrame, row + y, col + x, blockSize);

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
