#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/Reprojection.h"
#include "../include/MotionVectorEstimation.h"
#include <chrono>

using namespace std;
using Frame = std::vector<std::vector<std::vector<float>>>;



void writeFrames(const string& videoPath, int numberFrames, double newFps, const std::string& outputName){
    cout << "we are here" << endl;
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    std::cout << "Frame height: " << frameHeight << std::endl;
    std::cout << "Frame width: " << frameWidth << std::endl;

    cv::VideoWriter out(outputName + ".mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), newFps, cv::Size(frameWidth, frameHeight));

    cv::Mat prevFrame, currFrame, prevGray, currGray;
    cap.read(prevFrame);
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY, 1);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numberFrames && cap.read(currFrame); i++){
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY, 1);

        if (i > 0 && i % 2 == 0){

            MotionVectorEstimation mv;

            auto startMotion = std::chrono::high_resolution_clock::now();

            auto motionVectors = mv.estimateMotionVectors(prevGray, currGray);
            // auto motionVectors = mv.estimateAndRefineMotionVectors(prevFrame, currFrame, 16, 2, 4);

            auto endMotion = std::chrono::high_resolution_clock::now();
            auto durationMotion = std::chrono::duration_cast<std::chrono::milliseconds>(endMotion - startMotion);

            std::cout << "Motion vectors calculated" << std::endl;
            std::cout << "Motion vector Estimation took " << durationMotion.count() << " milliseconds." << std::endl;


            Reprojection rp;
            auto startReprojection = std::chrono::high_resolution_clock::now();

            auto reprojectedFrame = rp.reproject(currFrame, motionVectors, 16);

            // for (int i = 0; i < reprojectedFrame.rows; i++){
            // for ( int j = 0; i < reprojectedFrame.cols; j++){
            //     cout << "arbitrary channel values of prevFrame" << endl;
            //     cout << reprojectedFrame.at<cv::Vec3f>(i, j)[0] << ", " << reprojectedFrame.at<cv::Vec3f>(i, j)[1] <<", " << reprojectedFrame.at<cv::Vec3f>(i, j)[2]<<endl;
            // }
        // }

            auto endReprojection = std::chrono::high_resolution_clock::now();
            auto durationReprojection = std::chrono::duration_cast<std::chrono::milliseconds>(endReprojection - startReprojection);
            std::cout << "Reprojectionn took " << durationReprojection.count() << " milliseconds." << std::endl;
            

            out.write(reprojectedFrame);
            // cout << "Attempting to write" << std::endl;
            prevGray = currGray.clone();

        }
        else {
            out.write(currFrame);
            prevGray = currGray.clone();
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end- start);
    cout << "It took " << duration.count() << " milliseconds to write " << numberFrames << " frames" << endl;



    cap.release();
    out.release();

}

int main(){

    int numFrames = 100;
    auto start = std::chrono::high_resolution_clock::now();
    writeFrames("building_sample_video.mp4", numFrames, 30, "reprojected_buildings");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end- start);

    cout << "It took " << duration.count() << " milliseconds to write " << numFrames << " frames" << endl;
    return 0;
    
}