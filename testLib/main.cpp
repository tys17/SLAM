#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int main() {
    Mat img = imread("/home/jianwei/Downloads/RedEyeRemover/red_eyes.jpg");
    imwrite("/home/jianwei/code/SLAM/test/test.jpg", img);
    return 0;
}