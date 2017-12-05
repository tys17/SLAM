
//#include "stdafx.h"
#include <iostream>
#include <fstream>
#include<algorithm>
#include <sstream>
#include <stdio.h>
//#include <opencv.hpp>
//#include <core/core.hpp>
//#include <highgui/highgui.hpp>
//#include <features2d/features2d.hpp>
#include <windows.h>
#include<math.h>
#include<io.h>
#include <string>
#include <fstream>
#include <sstream>
#include <direct.h>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"

#include <Windows.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <strsafe.h>


#pragma comment(lib, "User32.lib")


using namespace cv;
using namespace std;

int RansacThreshold = 1;

vector<string> ReadIm()
{
    WIN32_FIND_DATA ffd;
    LARGE_INTEGER filesize;
    TCHAR szDir[MAX_PATH];
    size_t length_of_arg;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    vector<string> files;
    StringCchLength(".", MAX_PATH, &length_of_arg);


    StringCchCopy(szDir, MAX_PATH, ".");
    StringCchCat(szDir, MAX_PATH, TEXT("\\rgb\\*"));
    hFind = FindFirstFile(szDir, &ffd);


    do
    {
        if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            //_tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
        }
        else
        {
            string t = "./rgb/";
            filesize.LowPart = ffd.nFileSizeLow;
            filesize.HighPart = ffd.nFileSizeHigh;
            //_tprintf(TEXT("  %s   %ld bytes\n"), ffd.cFileName, filesize.QuadPart);
            files.push_back(t.append(ffd.cFileName));
        }
    } while (FindNextFile(hFind, &ffd) != 0);

    FindClose(hFind);
    return files;
}



class Tracking
{
public:
    void ExtractORBFeature(Mat img, vector<Point2f> & points)
    {
        vector<KeyPoint> keypoints;
        Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
        orb->detect(img, keypoints);
        for (int i = 0; i < keypoints.size(); i++)
        {
            points.push_back(keypoints[i].pt);
        }
    }
    void LKCorrespondence(Mat img_1, Mat img_2, vector<Point2f> & points0, vector<Point2f> & points1)
    {
        Size subPixWinSize(10, 10), winSize(5, 5);
        TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 15, 0.1);
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(img_1, img_2, points0, points1, status, err, winSize,
                             3, termcrit);
        for (int i = points1.size() - 1; i >= 0; i--)
        {
            if (status[i] == 0)
            {
                points1.erase(points1.begin() + i, points1.begin() + i + 1);
                points0.erase(points0.begin() + i, points0.begin() + i + 1);
            }
        }
    }
    Mat Triangulate(vector<Point2f> & points0, vector<Point2f> & points1)
    {
        vector <uchar> RANSACStatus;
        Mat fundamental_matrix =
                findFundamentalMat(points0, points1, FM_RANSAC, RansacThreshold, 0.99, RANSACStatus);
        for (int i = points1.size() - 1; i >= 0; i--)
        {
            if (RANSACStatus[i] == 0)
            {
                points1.erase(points1.begin() + i, points1.begin() + i + 1);
                points0.erase(points0.begin() + i, points0.begin() + i + 1);
            }
        }
        return fundamental_matrix;
    }
    Mat defineIntrinsic()
    {
        double fx, fy, cx, cy;
        Mat K = cv::Mat::eye(3, 3, CV_64FC1);
        fx = 517.3;
        fy = 516.5;
        cx = 318.6;
        cy = 255.3;
        K.at<double>(0, 0) = fx;
        K.at<double>(1, 1) = fy;
        K.at<double>(0, 2) = cx;
        K.at<double>(1, 2) = cy;
        return K;
    }
    Mat MatrixToQuaternion(Mat R)
    {
        Mat q(4, 1, CV_64FC1);
        q.at<double>(0, 0) = sqrt(1 + R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2)) / 2;
        q.at<double>(1, 0) = (R.at<double>(1, 2) - R.at<double>(2, 1)) / (4 * q.at<double>(0, 0));
        q.at<double>(2, 0) = (R.at<double>(2, 0) - R.at<double>(0, 2)) / (4 * q.at<double>(0, 0));
        q.at<double>(3, 0) = (R.at<double>(0, 1) - R.at<double>(1, 0)) / (4 * q.at<double>(0, 0));
        return q;
    }
    Mat QuaternionToMatrix(Mat q)
    {
        Mat R(3, 3, CV_64FC1);
        R.at<double>(0, 0) = 1 - 2 * pow(q.at<double>(2, 0), 2) - 2 * pow(q.at<double>(3, 0), 2);
        R.at<double>(0, 1) = 2 * q.at<double>(1, 0)*q.at<double>(2, 0) + 2 * q.at<double>(0, 0)*q.at<double>(3, 0);
        R.at<double>(0, 2) = 2 * q.at<double>(1, 0)*q.at<double>(3, 0) - 2 * q.at<double>(0, 0)*q.at<double>(2, 0);
        R.at<double>(1, 0) = 2 * q.at<double>(1, 0)*q.at<double>(2, 0) - 2 * q.at<double>(0, 0)*q.at<double>(3, 0);
        R.at<double>(1, 1) = 1 - 2 * pow(q.at<double>(1, 0), 2) - 2 * pow(q.at<double>(3, 0), 2);
        R.at<double>(1, 2) = 2 * q.at<double>(2, 0)*q.at<double>(3, 0) + 2 * q.at<double>(0, 0)*q.at<double>(1, 0);
        R.at<double>(2, 0) = 2 * q.at<double>(1, 0)*q.at<double>(3, 0) + 2 * q.at<double>(0, 0)*q.at<double>(2, 0);
        R.at<double>(2, 1) = 2 * q.at<double>(2, 0)*q.at<double>(3, 0) - 2 * q.at<double>(0, 0)*q.at<double>(1, 0);
        R.at<double>(2, 2) = 1 - 2 * pow(q.at<double>(1, 0), 2) - 2 * pow(q.at<double>(2, 0), 2);
        return R;
    }

};
int main(int argc, char** argv)
{
    vector <string> files=ReadIm();
    int interval = 1;
    int i = 0;
    vector<Mat> Orientations, poss, Rs;

    Mat BenchMarkFrame= imread(files[i]);
    Mat PreFrame, CurFrame;
    vector<Point2f> BenchMarkPoints;
    Ptr<Tracking> tracker;
    tracker->ExtractORBFeature(BenchMarkFrame, BenchMarkPoints);
    vector<Point2f> CurPoints(BenchMarkPoints.size()), PrePoints;
    copy(BenchMarkPoints.begin(), BenchMarkPoints.end(), CurPoints.begin());
    BenchMarkFrame.copyTo(CurFrame);

    Mat K = tracker->defineIntrinsic();
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
    CurR = tracker->QuaternionToMatrix(CurOrientation).t();
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
        tracker->LKCorrespondence(PreFrame, CurFrame, PrePoints, CurPoints);


        Mat fundamental_matrix = tracker->Triangulate(PrePoints, CurPoints);


        Mat E = Kt*fundamental_matrix*K;
        Mat TemptR, TemptT;
        Curpos.copyTo(Prepos);
        CurOrientation.copyTo(PreOrientation);
        CurR.copyTo(PreR);
        CurT.copyTo(PreT);
        recoverPose(E, PrePoints, CurPoints, TemptR, TemptT, focal);
        CurR = TemptR*PreR;
        CurOrientation = tracker->MatrixToQuaternion(CurR.t());
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
            tracker->ExtractORBFeature(BenchMarkFrame, BenchMarkPoints);
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
        Mat Tempt = tracker->QuaternionToMatrix(Orientations[j]);
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
    return 0;
}