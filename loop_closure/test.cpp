//
// Created by root on 12/4/17.
//

#include <iostream>

// opencv
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/xfeatures2d.hpp"

// DBoW3
#include "DBoW3/DBoW3.h"

//ceres-solver
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "fileIO.h"
#include "Tracking.h"
#include "Database.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(){
    fileIO fIO;
    vector <string> files= fIO.loadImages("/home/jianwei/code/SLAM/data/rgb");
    cv::Mat frame0 = imread(files[0]);
    cv::Mat frame1 = imread(files[1]);
    Ptr<ORB> detector = ORB::create();
    vector<KeyPoint> k0, k1;
    Mat d0, d1;
    detector->detectAndCompute(frame0, Mat(), k0, d0);
    detector->detectAndCompute(frame1, Mat(), k1, d1);
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    d0.convertTo(d0, CV_32F);
    d1.convertTo(d1, CV_32F);
    matcher.match(d0, d1, matches);
    cout << matches.size() << endl;
    return 0;
}
