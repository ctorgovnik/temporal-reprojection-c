#include "../include/Reprojection.h"
#include <array>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <thread>
#include <functional>

using namespace std;

vector<array<int, 2>> Reprojection::findNonZeroNeighbors(const cv::Mat& frame, int y, int x){

    int height = frame.rows;
    int width = frame.cols;
    vector<array<int, 2>> nonZeroNeighbors(4, {0, 0}); // Pre-fill with default values

    // Directions: topLeft, topRight, bottomLeft, bottomRight
    vector<pair<int, int>> directions = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (int i = 0; i < 4; i++) {
        int dy = directions[i].first;
        int dx = directions[i].second;
        int ny = y + dy;
        int nx = x + dx;

        while (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            if (!(frame.at<cv::Vec3b>(ny, nx)[0] == 0 && frame.at<cv::Vec3b>(ny, nx)[1] == 0 && frame.at<cv::Vec3b>(ny, nx)[2] == 0)) {
                nonZeroNeighbors[i] = {ny - y, nx - x};
                break;
            }
            ny += dy;
            nx += dx;
        }
    }
    return nonZeroNeighbors;
}


float Reprojection::safeInverse(float distance, float epsilon){

    if (distance == 0){
        return 1 / epsilon;
    }
    else {
        return 1 / distance;
    }

}

float Reprojection::gaussianWeight(float distance, float sigma) {
    return exp(-0.5 * pow(distance / sigma, 2));
}


cv::Vec3b Reprojection::bilinearInterpolation(
    const cv::Mat& frame, 
    int y, int x, 
    const std::array<int, 2>& topLeft, 
    const std::array<int, 2>& topRight, 
    const std::array<int, 2>& bottomLeft,
    const std::array<int, 2>& bottomRight){

    // Step 1: Calculate distances
    // Assuming distance calculation is done prior to this function call

    float sigma = 1.0;
    // Step 2: Compute inverse distance weights
    float topLeftWeight = gaussianWeight(sqrt(pow(topLeft[0], 2) + pow(topLeft[1], 2)), sigma);
    float topRightWeight = gaussianWeight(sqrt(pow(topRight[0], 2) + pow(topRight[1], 2)), sigma);
    float bottomLeftWeight = gaussianWeight(sqrt(pow(bottomLeft[0], 2) + pow(bottomLeft[1], 2)), sigma);
    float bottomRightWeight = gaussianWeight(sqrt(pow(bottomRight[0], 2) + pow(bottomRight[1], 2)), sigma);

    // Step 3: Normalize weights
    float totalWeight = topLeftWeight + topRightWeight + bottomLeftWeight + bottomRightWeight;
    topLeftWeight /= totalWeight;
    topRightWeight /= totalWeight;
    bottomLeftWeight /= totalWeight;
    bottomRightWeight /= totalWeight;

    // Step 4: Weighted sum of neighbors' pixel values
    cv::Vec3b interpolatedPixel = cv::Vec3b(0,0,0);
    for (int i = 0; i < 3; ++i) { // For each channel
        float weightedSum = 
            topLeftWeight * frame.at<cv::Vec3b>(y + topLeft[0], x + topLeft[1])[i] +
            topRightWeight * frame.at<cv::Vec3b>(y + topRight[0], x + topRight[1])[i] +
            bottomLeftWeight * frame.at<cv::Vec3b>(y + bottomLeft[0], x + bottomLeft[1])[i] +
            bottomRightWeight * frame.at<cv::Vec3b>(y + bottomRight[0], x + bottomRight[1])[i];

        // Step 5: Assign interpolated value
        interpolatedPixel[i] = cv::saturate_cast<uchar>(weightedSum);
    }

    return interpolatedPixel;
}

cv::Vec3b Reprojection::interpolateFromKernel(const cv::Mat& frame, int y, int x, int kernelSize = 3){

    int halfSize = kernelSize / 2;

    vector<cv::Vec3b> nonZeroPixels;
    // Define bounds for the kernel
    int startY, endY, startX, endX;

    while(nonZeroPixels.size() < 4){
        startY = (0, y - halfSize);
        endY = min(frame.rows, y + halfSize + 1);
        startX = max(0, x - halfSize);
        endX = min(frame.cols, x + halfSize + 1);

        // check if out of bounds, shift kernel
        for (int row = startY; row < endY; row++){
            for (int col = startX; col < endX; col++){

                cv::Vec3b pixel = frame.at<cv::Vec3b>(row, col);
                if (!(pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)) {
                    nonZeroPixels.push_back(pixel);
                }

            }

           
        }

         halfSize += 1;
        // if loops again, increase kernel size, can do this at end of loop

    }

    // Compute the average of the non-zero pixels
    long sum[3] = {0, 0, 0};
    for (const auto& p : nonZeroPixels) {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    int count = nonZeroPixels.size();
    return cv::Vec3b(sum[0] / count, sum[1] / count, sum[2] / count);
    

}

void Reprojection::processBlock(cv::Mat& frame, int startRow, int endRow, int width){

    for (int row = startRow; row < endRow; row++) {
        for (int col = 0; col < width; col++) {
            if (frame.at<cv::Vec3b>(row, col) == cv::Vec3b(0,0,0)) {
                frame.at<cv::Vec3b>(row,col) = interpolateFromKernel(frame, row, col);
            }
        }
    }
}

void Reprojection::fillInZeros(cv::Mat& frame){

    int height = frame.rows;
    int width = frame.cols;

    int numThreads = 24;
    vector<thread> threads;

    // split image into horizontal blocks
    int blockSize = height / numThreads;

    for (int i = 0; i < numThreads; i++){
        int startRow = i*blockSize;
        int endRow = (i + 1) * blockSize;
        if (i == numThreads - 1) endRow = height;

        threads.push_back(thread([this, &frame, startRow, endRow, width]() {
            this->processBlock(frame, startRow, endRow, width);
        }));

    }

    for (auto& t : threads){

        t.join();

    }

}

void checkZeros(const cv::Mat& frame) {
    // Ensure that we are dealing with a 3-channel image
    if (frame.channels() != 3) {
        std::cout << "The image is not a 3-channel image." << std::endl;
        return;
    }

    int zeroCount = 0;

    // Iterate through each pixel in the image
    for (int row = 0; row < frame.rows; ++row) {
        for (int col = 0; col < frame.cols; ++col) {
            // Access the pixel at (row, col)
            const cv::Vec3b& pixel = frame.at<cv::Vec3b>(row, col);

            // Check if all three channels of the pixel are zero
            if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
               
                zeroCount += 1;
            }
        }
    }

    cout << "count: " << zeroCount << endl;
}

cv::Mat Reprojection::reproject(const cv::Mat& lastFrame, 
        const std::vector<std::vector<MotionVector>>& motionVectors, 
        int blockSize){

            cv::Mat newFrame = cv::Mat::zeros(lastFrame.size(), lastFrame.type());

            

            
            for (int row = 0; row < motionVectors.size(); row++){
                for (int col = 0; col < motionVectors[0].size(); col++){

                    if (row >= motionVectors.size() || col >= motionVectors[0].size()){
                        continue;
                    }
                    const MotionVector& mv = motionVectors[row][col];

                    auto start = get<0>(mv);
                    auto displacement = get<1>(mv);

                    int startY = start.first, startX = start.second;
                    int endY = startY + displacement.first, endX = startX + displacement.second;

                    if (0 <= endY && endY < lastFrame.rows - blockSize && 0 <= endX && endX < lastFrame.cols - blockSize
                       && col * blockSize < lastFrame.cols && row * blockSize < lastFrame.rows){
                        
            

                        cv::Rect srcRect(col * blockSize, row * blockSize, blockSize, blockSize);
                        cv::Rect dstRect(endX, endY, blockSize, blockSize);

                        dstRect = dstRect & cv::Rect(0, 0, lastFrame.cols, lastFrame.rows);

                        if (0 <= srcRect.x && 0 <= srcRect.y &&
                        srcRect.x + srcRect.width < lastFrame.cols &&
                        srcRect.y + srcRect.height < lastFrame.rows) {
                        // Ensure dstRect is fully contained within newFrame before adjusting
                            if (0 <= dstRect.x && 0 <= dstRect.y &&
                                dstRect.x + dstRect.width < newFrame.cols &&
                                dstRect.y + dstRect.height < newFrame.rows) {
                                // Perform intersection with newFrame bounds to ensure containment
                                dstRect = dstRect & cv::Rect(0, 0, newFrame.cols, newFrame.rows);

                            // Now safe to perform the copy operation
                                lastFrame(srcRect).copyTo(newFrame(dstRect));
                            } else {
                                cout << "dstRect out of bounds of newFrame" << endl;
                                continue;
                            }
                        } else {
                            cout << "srcRect out of bounds of lastFrame" << endl;
                            continue;
                        }


                        
                        }
                    }

                }
            // checkZeros(newFrame);
            fillInZeros(newFrame);
            // checkZeros(newFrame);


            return newFrame;


        }

