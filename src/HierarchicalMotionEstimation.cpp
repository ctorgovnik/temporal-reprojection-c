#include "../include/HierarchicalMotionEstimation.h"

using namespace std;

HierarchicalMotionEstimation::HierarchicalMotionEstimation() {
    // Constructor
}

cv::Mat HierarchicalMotionEstimation::reprojectImage(const cv::Mat& image, const std::vector<std::vector<std::pair<int, int>>>& motionVectors) {
    cv::Mat reprojectedImage = cv::Mat::zeros(image.size(), image.type());

    for (int y = 0; y < motionVectors.size(); ++y) {
        for (int x = 0; x < motionVectors[y].size(); ++x) {
            auto [dy, dx] = motionVectors[y][x]; // Get the motion vector for the current pixel/block

            int newX = x + dx;
            int newY = y + dy;

            cout << "new position: " << newX << ", " << newY << endl;


            // Check if the new position is within the bounds of the image
            if (newX >= 0 && newX < image.cols && newY >= 0 && newY < image.rows) {
                // For simplicity, directly copy the pixel value. For blocks, you'd copy the entire block.
                reprojectedImage.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x);
            }
        }
    }

    // Optional: Fill in gaps created by the reprojection, e.g., using interpolation
    // This step is important for a smooth, artifact-free image but is omitted here for brevity

    return reprojectedImage;
}


std::vector<std::vector<std::pair<int, int>>> HierarchicalMotionEstimation::estimateMotionVectors(
    const cv::Mat& prevFrame, 
    const cv::Mat& currFrame, 
    int blockSize, 
    int searchRange) {
    
    // Step 1: Build resolution pyramids for previous and current frames
    int levels = 4; 
    auto prevPyramid = buildResolutionPyramid(prevFrame, levels);
    auto currPyramid = buildResolutionPyramid(currFrame, levels);

    std::vector<std::vector<std::vector<std::pair<int, int>>>> motionVectorsPerLevel(levels);

    // Initialize motion vectors for the coarsest level
    motionVectorsPerLevel[levels - 1] = initializeMotionVectorsForLevel(prevPyramid[levels - 1], blockSize);

    // Estimate motion starting from the second coarsest level to the finest
    for (int level = levels - 2; level >= 0; --level) {
        // Warp the coarser previous frame towards the current frame based on coarse level vectors
        cv::Mat warpedPrev = warpImage(prevPyramid[level + 1], motionVectorsPerLevel[level + 1], blockSize);

        // Now, refine motion vectors at the current level using the warped previous frame
        motionVectorsPerLevel[level] = refineMotionVectorsAtLevel(
            warpedPrev, currPyramid[level], 
            motionVectorsPerLevel[level + 1], blockSize, searchRange);
    }

    // Return finest level motion vectors
    return motionVectorsPerLevel[0];
}

cv::Mat HierarchicalMotionEstimation::getBlock(const cv::Mat& frame, int startRow, int startCol, int blockSize){
    int effectiveBlockSizeWidth = std::min(blockSize, frame.cols - startCol);
    int effectiveBlockSizeHeight = std::min(blockSize, frame.rows - startRow);
    // std::cout << "StartRow: " << startRow << ", StartCol: " << startCol << ", BlockSize: " << blockSize << std::endl;
    // std::cout << "Frame dimensions: " << frame.rows << "x" << frame.cols << std::endl;

    cv::Rect roi(startCol, startRow, effectiveBlockSizeWidth, effectiveBlockSizeHeight);
    return frame(roi);
}

std::vector<std::vector<std::pair<int, int>>> HierarchicalMotionEstimation::initializeMotionVectorsForLevel(
    const cv::Mat& frame, int blockSize) {
    int rows = std::ceil(static_cast<float>(frame.rows) / blockSize);
    int cols = std::ceil(static_cast<float>(frame.cols) / blockSize);

    std::vector<std::vector<std::pair<int, int>>> motionVectors(rows, std::vector<std::pair<int, int>>(cols, {0, 0}));

    // If you have a heuristic for initial motion, apply it here
    // For example, assuming a slight motion to the right:
    // for (auto& row : motionVectors)
    //     for (auto& vec : row)
    //         vec = {0, 1}; // Slight movement to the right

    return motionVectors;
}

std::vector<cv::Mat> HierarchicalMotionEstimation::buildResolutionPyramid(const cv::Mat& frame, int levels) {
    // initialize pyramid as vector of images
    vector<cv::Mat> pyramid;
    cv::Mat currentLevel = frame;

    // original frame as lowest level of pyramid
    pyramid.push_back(currentLevel); 

    for (int i = 0; i < levels; i++){
        cv::Mat reduced;
        cv::pyrDown(currentLevel, reduced);
        pyramid.push_back(reduced);
        currentLevel = reduced;
    }

    // make first element the top level of pyramid
    std::reverse(pyramid.begin(), pyramid.end());

    return pyramid;
}

cv::Mat HierarchicalMotionEstimation::warpImage(const cv::Mat& image, const std::vector<std::vector<std::pair<int, int>>>& motionVectors, int blockSize) {
    cv::Mat warpedImage = image.clone(); // Use clone() to copy the input image structure

    // Iterate through each motion vector
    for (int y = 0; y < motionVectors.size(); y++) {
        for (int x = 0; x < motionVectors[y].size(); x++) {
            auto vector = motionVectors[y][x];
            // Calculate source and destination rectangles based on the motion vector
            cv::Rect srcRect(x * blockSize, y * blockSize, blockSize, blockSize);
            int destX = x * blockSize + vector.second * blockSize; // Adjust by blockSize for correct level
            int destY = y * blockSize + vector.first * blockSize; // Adjust by blockSize for correct level
            cv::Rect dstRect(destX, destY, blockSize, blockSize);

            // Perform the warping if the destination is within image bounds
            if ((dstRect & cv::Rect(0, 0, warpedImage.cols, warpedImage.rows)) == dstRect) {
                // This may require additional logic to handle partial overlaps or extrapolation at boundaries
                warpedImage(srcRect).copyTo(warpedImage(dstRect));
            }
        }
    }

    return warpedImage;
}

float HierarchicalMotionEstimation::computePatchDistance(const cv::Mat& patch1, const cv::Mat& patch2) {
    cv::Mat diff;
    cv::absdiff(patch1, patch2, diff); // Compute absolute difference
    cv::Scalar sum = cv::sum(diff); // Sum the differences

    int totalPixels = patch1.rows * patch1.cols;
    return sum[0] / totalPixels;
}

std::vector<std::vector<std::pair<int, int>>> HierarchicalMotionEstimation::refineMotionVectorsAtLevel(
    const cv::Mat& prevLevel, 
    const cv::Mat& currLevel, 
    const std::vector<std::vector<std::pair<int, int>>>& coarseMotionVectors, 
    int blockSize, 
    int searchRange) {
    int rows = std::ceil(static_cast<float>(currLevel.rows) / blockSize);
    int cols = std::ceil(static_cast<float>(currLevel.cols) / blockSize);
    
    std::vector<std::vector<std::pair<int, int>>> refinedVectors(rows, std::vector<std::pair<int, int>>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            auto [dy, dx] = coarseMotionVectors[i][j]; // Coarse vector
            float bestMad = std::numeric_limits<float>::infinity();
            std::pair<int, int> bestVector = {0, 0};

            for (int y = dy - searchRange; y <= dy + searchRange; ++y) {
                for (int x = dx - searchRange; x <= dx + searchRange; ++x) {
                    // Ensure the search window is within the frame boundaries
                    if (i * blockSize + y < 0 || j * blockSize + x < 0 || 
                        i * blockSize + y + blockSize > prevLevel.rows || 
                        j * blockSize + x + blockSize > prevLevel.cols) continue;

                    cv::Mat prevBlock = getBlock(prevLevel, i * blockSize + y, j * blockSize + x, blockSize);
                    cv::Mat currBlock = getBlock(currLevel, i * blockSize, j * blockSize, blockSize);

                    if (prevBlock.rows != blockSize || prevBlock.cols != blockSize || 
                        currBlock.rows != blockSize || currBlock.cols != blockSize) continue;

                    float mad = computePatchDistance(prevBlock, currBlock);
                    if (mad < bestMad) {
                        bestMad = mad;
                        bestVector = {y, x};
                    }
                }
            }
            
            refinedVectors[i][j] = bestVector;
        }
    }
    
    return refinedVectors;
}


