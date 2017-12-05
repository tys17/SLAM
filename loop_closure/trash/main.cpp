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

// DBoW3
#include "DBoW3/DBoW3.h"

//ceres-solver
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "fileIO.h"
#include "Tracking.h"

using namespace std;
using namespace cv;
using namespace DBoW3;

#define PIX_THREH 2


int main() {
    //Vocabulary voc("/home/jianwei/code/SLAM/loop_closure/orbvoc.dbow3");
    //std::cout << "Hello, World!" << std::endl;
    fileIO test;
    double RansacThreshold = PIX_THREH;
    vector<string> files = test.loadImages("/home/jianwei/code/SLAM/data/rgb");

    int interval = 1;
    int i = 0;
    vector<Mat> Orientations, poss, Rs;

    Mat BenchMarkFrame= imread(files[i]);
    Mat PreFrame, CurFrame;
    vector<Point2f> BenchMarkPoints;
    vector<KeyPoint> BenchMarkkeypoints;
    Mat BenchMarkDescriptors;
    //Ptr<Tracking> tracker;
    Tracking tracker;
    tracker.RansacThreshold = PIX_THREH;
    tracker.ExtractORBFeature(BenchMarkFrame, BenchMarkPoints, BenchMarkkeypoints, BenchMarkDescriptors);
    //tracker->ExtractORBFeature(BenchMarkFrame, BenchMarkPoints);
    vector<Point2f> CurPoints(BenchMarkPoints.size()), PrePoints;
    copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
    BenchMarkFrame.copyTo(CurFrame);

    //Mat K = tracker->defineIntrinsic();
    Mat K = tracker.defineIntrinsic();
    Mat Kt = K.t();
    i += interval;
    double focal = 517;
    Mat CurOrientation(4, 1, CV_64FC1), PreOrientation(4, 1, CV_64FC1), Curpos(3, 1, CV_64FC1), Prepos(3, 1, CV_64FC1);
    Mat CurR(3, 3, CV_64FC1), PreR(3, 3, CV_64FC1), CurT(3, 1, CV_64FC1), PreT(3, 1, CV_64FC1);
    //Initialization
    CurOrientation.at<double>(0, 0) = 0.8596;
    CurOrientation.at<double>(1, 0) = -0.3534;
    CurOrientation.at<double>(2, 0) = 0.0838;
    CurOrientation.at<double>(3, 0) = -0.3594;
    Curpos.at<double>(0, 0) = 0.4388;
    Curpos.at<double>(1, 0) = -0.4332;
    Curpos.at<double>(2, 0) = 1.4779;
    //CurR = tracker->QuaternionToMatrix(CurOrientation).t();
    CurR = tracker.QuaternionToMatrix(CurOrientation).t();
    CurT = -1 * CurR*Curpos;


    Orientations.push_back(CurOrientation);
    //Rs.push_back(CurR);
    Mat tempt(1, 3, CV_64FC1);
    tempt = Curpos.clone();
    poss.push_back(tempt);
    //poss.push_back(Curpos);
    //int q = 1;
    while (i < files.size())
    {
        CurFrame.copyTo(PreFrame);
        CurFrame = imread(files[i]);
        PrePoints.clear();
        PrePoints.resize(CurPoints.size());
        copy(CurPoints.begin(), CurPoints.end(), PrePoints.begin());
        //Check if CurPoints could have a size nonzero
        //tracker->LKCorrespondence(PreFrame, CurFrame, PrePoints, CurPoints);
        tracker.LKCorrespondence(PreFrame, CurFrame, PrePoints, CurPoints);


        //Mat fundamental_matrix = tracker->Triangulate(PrePoints, CurPoints);
        Mat fundamental_matrix = tracker.Triangulate(PrePoints, CurPoints);


        Mat E = Kt*fundamental_matrix*K;
        Mat TemptR, TemptT;
        Curpos.copyTo(Prepos);
        CurOrientation.copyTo(PreOrientation);
        CurR.copyTo(PreR);
        CurT.copyTo(PreT);
        recoverPose(E, PrePoints, CurPoints, TemptR, TemptT, focal);
        CurR = TemptR*PreR;
        //CurOrientation = tracker->MatrixToQuaternion(CurR.t());
        CurOrientation = tracker.MatrixToQuaternion(CurR.t());
        Curpos = -1 * PreR.inv()*TemptR.inv()*TemptT - PreR.inv()*PreT;
        CurT =-1* CurR*Curpos;
        /*cout << Curpos  << endl;*/
        //CurT = TemptT + PreT;
        //Curpos = -1 * CurR.inv()*CurT;

        double PointsRatio = double(CurPoints.size()) / double(BenchMarkPoints.size());
        if (PointsRatio < 0.6)
        {
            //size changes
            BenchMarkPoints.clear();
            CurFrame.copyTo(BenchMarkFrame);
            //tracker->ExtractORBFeature(BenchMarkFrame, BenchMarkPoints);
            tracker.ExtractORBFeature(BenchMarkFrame, BenchMarkPoints, BenchMarkkeypoints, BenchMarkDescriptors);
            // to be modified here, remember to store (deep copy) the keypoints and descriptors
            CurPoints.clear();
            CurPoints.resize(BenchMarkPoints.size());
            copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
        }
        Orientations.push_back(CurOrientation);
        Mat tempt(1, 3, CV_64FC1);
        tempt=Curpos.clone();
        poss.push_back(tempt);
        //std::cout<< poss.back().at<double>(0, 0) << " " << poss.back().at<double>(1, 0) << " " << poss.back().at<double>(2, 0) << " "<<endl;
        //cout << poss.back() << endl;
        //std::cout << poss[0].at<double>(0, 0) << " " << poss[0].at<double>(1, 0) << " " << poss[0].at<double>(2, 0) << " " << endl;
        //q++;
        i += interval;
    }

    ofstream location_out;
    string filename;
    filename = "Interval=" + to_string(interval) + "_Ransac=" + to_string(RansacThreshold) + ".txt";
    location_out.open(filename, std::ios::out | std::ios::app);
    if (!location_out.is_open())
        return 0;
    int num = Orientations.size();
    location_out << num << endl;
    for (int j = 0; j < num; j++)
    {
        //Mat Tempt = tracker->QuaternionToMatrix(Orientations[j]);
        Mat Tempt = tracker.QuaternionToMatrix(Orientations[j]);
        location_out << K.at<double>(0, 0) << " " << K.at<double>(0, 1) << " " << K.at<double>(0, 2) << " " << K.at<double>(1, 0) << " " << K.at<double>(1, 1) << " " << K.at<double>(1, 2) << " " << K.at<double>(2, 0) << " " << K.at<double>(2, 1) << " " << K.at<double>(2, 2) << " ";
        location_out << Tempt.at<double>(0, 0) << " " << Tempt.at<double>(0, 1) << " " << Tempt.at<double>(0, 2) << " " << Tempt.at<double>(1, 0) << " " << Tempt.at<double>(1, 1) << " " << Tempt.at<double>(1, 2) << " " << Tempt.at<double>(2, 0) << " " << Tempt.at<double>(2, 1) << " " << Tempt.at<double>(2, 2) << " ";
        location_out << poss[j].at<double>(0, 0) << " " << poss[j].at<double>(1, 0) << " " << poss[j].at<double>(2, 0) << " ";
        /*cout << poss[0].at<double>(0, 0) << " " << poss[0].at<double>(1, 0) << " " << poss[0].at<double>(2, 0) << " " << endl;*/
        //cout << poss[1].at<double>(0, 0) << " " << poss[1].at<double>(1, 0) << " " << poss[1].at<double>(2, 0) << " " << endl;
        //cout << poss[2].at<double>(0, 0) << " " << poss[2].at<double>(1, 0) << " " << poss[2].at<double>(2, 0) << " " << endl;
        //cout << poss[3].at<double>(0, 0) << " " << poss[3].at<double>(1, 0) << " " << poss[3].at<double>(2, 0) << " " << endl;
        location_out << 640 << " " << 480 << endl;
    }

    location_out.close();
    //cout << fileList.size() << endl;
    return 0;
}