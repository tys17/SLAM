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
#include "Database.h"

using namespace std;
using namespace cv;


#define PIX_THREH 2


vector<KeyPoint> convertToKeyPoints(vector<Point2f>& data){
    vector<KeyPoint> res;
    for (size_t i = 0; i < data.size(); i++) {
        res.push_back(cv::KeyPoint(data[i], 1.f));
    }
    return res;
}


int main() {
    double RansacThreshold = PIX_THREH;
    //DBoW3::Vocabulary voc("/home/jianwei/code/SLAM/loop_closure/orbvoc.dbow3");
    //DBoW3::Vocabulary voc("/home/jianwei/code/SLAM/loop_closure/built_voc.yml.gz");
    DBoW3::Vocabulary voc("/home/jianwei/code/SLAM/loop_closure/test.dbow3");
    Database db(voc);

    int debug_LoopCount = 0;

    fileIO fIO;
    vector <string> files=fIO.loadImages("/home/jianwei/code/SLAM/data/rgb");
    int interval = 1;
    int i = 0;
    vector<Mat> Orientations, poss, Rs;

    Mat BenchMarkFrame= imread(files[i]);

    //For Iphone Image
    //Rect Template(161, 0, 960, 720);
    //imshow("0", BenchMarkFrame);
    //CurFrame = CurFrame(Template);
    resize(BenchMarkFrame, BenchMarkFrame, Size(640, 480));
    //Mat gray;
    //cvtColor(BenchMarkFrame, gray, COLOR_BGR2GRAY);
    //imshow("1", gray);
    //waitKey(0);


    Mat PreFrame, CurFrame;
    vector<Point2f> BenchMarkPoints;
    Tracking tracker;
    tracker.RansacThreshold = PIX_THREH;
    Mat CurDescriptors;

    tracker.ExtractORBFeature(BenchMarkFrame, BenchMarkPoints, CurDescriptors);
    vector<Point2f> CurPoints(BenchMarkPoints.size()), PrePoints;
    copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
    BenchMarkFrame.copyTo(CurFrame);

    Mat K = tracker.defineIntrinsic();
    Mat Kt = K.t();
    i += interval;
    double focal = 517;
    Mat CurOrientation(4, 1, CV_64FC1), PreOrientation(4, 1, CV_64FC1), Curpos(3, 1, CV_64FC1), Prepos(3, 1, CV_64FC1);
    Mat CurR(3, 3, CV_64FC1), PreR(3, 3, CV_64FC1), CurT(3, 1, CV_64FC1), PreT(3, 1, CV_64FC1);
    //Initialization For RGB Dataset
    CurOrientation.at<double>(0, 0) = 0.8596;
    CurOrientation.at<double>(1, 0) = -0.3534;
    CurOrientation.at<double>(2, 0) = 0.0838;
    CurOrientation.at<double>(3, 0) = -0.3594;
    Curpos.at<double>(0, 0) = 0.4388;
    Curpos.at<double>(1, 0) = -0.4332;
    Curpos.at<double>(2, 0) = 1.4779;

    //Initialization For Iphone
    /*CurOrientation.at<double>(0, 0) = 1;
    CurOrientation.at<double>(1, 0) = 0;
    CurOrientation.at<double>(2, 0) = 0;
    CurOrientation.at<double>(3, 0) = 0;
    Curpos.at<double>(0, 0) = 0;
    Curpos.at<double>(1, 0) = 0;
    Curpos.at<double>(2, 0) = 0;*/

    CurR = tracker.QuaternionToMatrix(CurOrientation).t();
    CurT = -1 * CurR*Curpos;


    Orientations.push_back(CurOrientation);
    //Rs.push_back(CurR);
    Mat tempt(1, 3, CV_64FC1);
    tempt = Curpos.clone();
    poss.push_back(tempt);
    //poss.push_back(Curpos);
    //int q = 1;
    int BenchMarkFlag = 1;

    /*// convert point2f to keypoints
    vector<KeyPoint> CurKeyPoints = convertToKeyPoints(CurPoints);
    Mat CurDescriptors;
    Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
    orb->compute(CurFrame, CurKeyPoints, CurDescriptors);*/

    // build a keyframe
    vector<int> feature_ids;
    int CurFrameID = 0;

    cv::Ptr<cv::Feature2D> fdetector;
    fdetector=cv::ORB::create();
    Mat debug_CurDescriptors;
    vector<KeyPoint> debug_KeyPoints;
    fdetector->detectAndCompute(CurFrame, cv::Mat(), debug_KeyPoints, debug_CurDescriptors);
    //KeyFrame keyframe(feature_ids, CurDescriptors, CurFrameID, CurR, CurT);
    KeyFrame keyframe(feature_ids, debug_CurDescriptors, CurFrameID, CurR, CurT);
    CurFrameID++;

    // add a keyframe to database
    db.addKeyframe(keyframe);

    while (i < files.size())
    {
        CurFrame.copyTo(PreFrame);
        CurFrame = imread(files[i]);
        resize(CurFrame, CurFrame, Size(640, 480));

        PrePoints.clear();
        PrePoints.resize(CurPoints.size());
        copy(CurPoints.begin(), CurPoints.end(), PrePoints.begin());
        //Check if CurPoints could have a size nonzero

        tracker.LKCorrespondence(PreFrame, CurFrame, PrePoints, CurPoints);



        if (BenchMarkFlag == 1)
        {
            BenchMarkPoints.clear();
            BenchMarkPoints.resize(PrePoints.size());
            copy(PrePoints.begin(), PrePoints.end(), BenchMarkPoints.begin());
            BenchMarkFlag = 0;
            //cout << i << endl;
        }


        Mat fundamental_matrix = tracker.Triangulate(PrePoints, CurPoints);
        // Show the match between two adjacent frames



        //Show Epipolar Lines:
        if (1)
        {
            std::vector <cv::Vec3f> lines1;
            computeCorrespondEpilines(PrePoints, 1, fundamental_matrix, lines1);
            Mat Show;
            CurFrame.copyTo(Show);
            for (std::vector<cv::Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
            {
                // Draw the line between first and last column
                cv::line(Show,
                         cv::Point(0, -(*it)[2] / (*it)[1]),
                         cv::Point(Show.cols, -((*it)[2] +
                                                (*it)[0] * Show.cols) / (*it)[1]),
                         cv::Scalar(0, 255, 0));
            }
            imwrite(to_string(i)+".jpg", Show);;
        }

        Mat E = Kt*fundamental_matrix*K;
        Mat TemptR, TemptT, Ori_, Pos_relative;
        Curpos.copyTo(Prepos);
        CurOrientation.copyTo(PreOrientation);
        CurR.copyTo(PreR);
        CurT.copyTo(PreT);
        Point2d tmp(K.at<double>(0, 2), K.at<double>(1, 2));
        recoverPose(E, PrePoints, CurPoints, TemptR, TemptT, focal, tmp);
        //If the outputs are the orientation and the position.


        CurR = TemptR*PreR;
        CurOrientation = tracker.MatrixToQuaternion(CurR.t());
        Curpos = -1 * tracker.QuaternionToMatrix(PreOrientation)*TemptR.inv()*TemptT + Prepos;
        CurT =-1* CurR*Curpos;


        double PointsRatio = double(CurPoints.size()) / double(BenchMarkPoints.size());
        if (PointsRatio < 0.6|| CurPoints.size()<80)
        {
            //size changes
            BenchMarkPoints.clear();
            CurFrame.copyTo(BenchMarkFrame);
            //Mat CurDescriptors;
            tracker.ExtractORBFeature(BenchMarkFrame, BenchMarkPoints, CurDescriptors);
            CurPoints.clear();
            CurPoints.resize(BenchMarkPoints.size());
            copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
            BenchMarkFlag = 1;
        }
        Orientations.push_back(CurOrientation);
        Mat tempt(1, 3, CV_64FC1);
        tempt=Curpos.clone();
        poss.push_back(tempt);

        i += interval;

        /*// convert point2f to keypoints
        vector<KeyPoint> CurKeyPoints = convertToKeyPoints(CurPoints);

        Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
        orb->compute(CurFrame, CurKeyPoints, CurDescriptors);*/

        // build a keyframe
        //feature_ids
        fdetector->detectAndCompute(CurFrame, cv::Mat(), debug_KeyPoints, debug_CurDescriptors);
        //KeyFrame keyframe(feature_ids, CurDescriptors, CurFrameID, CurR, CurT);
        KeyFrame keyframe(feature_ids, debug_CurDescriptors, CurFrameID, CurR, CurT);
        //KeyFrame keyframe(feature_ids, CurDescriptors, CurFrameID, CurR, CurT);
        CurFrameID++;


        // detect loop closure
        int startID = db.detectLoop(keyframe);
        if (startID != -1){
            // loop detected
            cout << debug_LoopCount << ": " << startID << " and " << CurFrameID << endl;
            debug_LoopCount++;
        }

        // add a keyframe to database
        db.addKeyframe(keyframe);

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
        Mat Tempt = tracker.QuaternionToMatrix(Orientations[j]);
        location_out << K.at<double>(0, 0) << " " << K.at<double>(0, 1) << " " << K.at<double>(0, 2) << " " << K.at<double>(1, 0) << " " << K.at<double>(1, 1) << " " << K.at<double>(1, 2) << " " << K.at<double>(2, 0) << " " << K.at<double>(2, 1) << " " << K.at<double>(2, 2) << " ";
        location_out << Tempt.at<double>(0, 0) << " " << Tempt.at<double>(0, 1) << " " << Tempt.at<double>(0, 2) << " " << Tempt.at<double>(1, 0) << " " << Tempt.at<double>(1, 1) << " " << Tempt.at<double>(1, 2) << " " << Tempt.at<double>(2, 0) << " " << Tempt.at<double>(2, 1) << " " << Tempt.at<double>(2, 2) << " ";
        location_out << poss[j].at<double>(0, 0) << " " << poss[j].at<double>(1, 0) << " " << poss[j].at<double>(2, 0) << " ";
        location_out << 640 << " " << 480 << endl;
    }

    location_out.close();

    return 0;
}