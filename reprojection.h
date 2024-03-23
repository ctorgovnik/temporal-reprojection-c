#ifndef MOTION_VECTOR_ESTIMATION_H
#define REPROJECTION_H
#include <array>
#include <vector>
#include <tuple>

// Define the motion vector type for clarity
using MotionVector = std::tuple<std::pair<int, int>, std::pair<int, int>>; // ((x, y), (dx, dy))

class Reprojection{

private:

    // think about making a matrix variable as the 'new frame' to update throughout

    /**
     * given a pixel in a frame, find 4 nearest non-zero neighbors to that pixel
     * 
     * @param frame matrix of frame's pixels. height x width x 3
     * @param y y position of pixel in consideration
     * @param x x position of pixel in consideration
     * @return 4 coordinates corresponding to 4 nearest non-zero neighbors
    */
    std::vector<std::array<int, 2>> findNonZeroNeighbors(const std::vector<std::vector<std::vector<float>>>& frame, int y, int x);

    /**
     * computes safe inverse of distance, avoiding division by zero
     * 
     * @param distance distance value
     * @param epsilon small value close to 0 to replace division by zero
     * @return inverse of distance without dividing by zero
    */
    float safeInverse(float distance, float epsilon=1e-10);

    /**
     * perform bilinear interpolation on a zeroed-out pixel (x, y), using nearest non-zero neighbors, each weighted by inverse of their respective distance to the pixel
     * 
     * @param frame matrix of pixels. height x width x 3
     * @param y y position of pixel in consideration
     * @param x x position of pixel in consideration
     * @param topLeft coordinate of top left non-zero neighbor. Array of dim 2: (y, x)
     * @param topRight coordinate of top right non-zero neighbor. Array of dim 2: (y, x)
     * @param bottomLeft coordinate of bottom left non-zero neighbor. Array of dim 2: (y, x)
     * @param bottomRight coordinate of bottom right non-zero neighbor. Array of dim 2: (y, x)
     * @return interpolated pixel with non-zero value
    */
    std::array<float, 3> bilinearInterpolation(
        const std::vector<std::vector<std::vector<float>>>& frame, 
        int y,
        int x, 
        const std::array<int, 2>& topLeft, 
        const std::array<int, 2>& topRight, 
        const std::array<int, 2>& bottomLeft,
        const std::array<int, 2>& bottomRight
    );

    /**
     * fills in all zero pixels by performing bilinear interpolation
     * 
     * @param frame matrix of frame's pixels. height x width x 3
     * @return a frame where every pixel has a non-zero value
    */
    std::vector<std::vector<std::vector<float>>> fillInZeros(const std::vector<std::vector<std::vector<float>>>& frame);


public:

    /**
     * given a frame and motion vectors, reproject frame's pixels to create a new frame
     * 
     * @param lastFrame matrix of last frame's pixels. height x width x 3
     * @param motionVectors matrix of motionVectors corresponding to lastFrame's movement. height x width x 2
     * @param blockSize size of each block. partitions matrix into blockSize x blockSize x 3 blocks
     * @return a new frame from warping past frame's pixels
    */
    std::vector<std::vector<std::vector<float>>> reproject(
        const std::vector<std::vector<std::vector<float>>>& lastFrame, 
        const std::vector<std::vector<MotionVector>>& motionVectors, 
        int blockSize
    );

};

#endif