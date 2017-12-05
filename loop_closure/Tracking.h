//
// Created by root on 12/3/17.
//

#ifndef LOOP_CLOSURE_TRACKING_H
#define LOOP_CLOSURE_TRACKING_H

#include <iostream>
#include <fstream>
#include<algorithm>
#include <string>
#include <vector>
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

using namespace std;
using namespace cv;

class Tracking {
public:
    double RansacThreshold = 1;

    void ExtractORBFeature(Mat img, vector<Point2f> & points, Mat& descriptors)
    {
        vector<KeyPoint> keypoints;
        Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
        orb->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        //orb->detect(img, keypoints);
        for (int i = 0; i < keypoints.size(); i++)
        {
            points.push_back(keypoints[i].pt);
        }
    }

    /*void ExtractORBFeature(Mat img, vector<Point2f> & points)
    {
        vector<KeyPoint> keypoints;
        Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
        orb->detect(img, keypoints);
        for (int i = 0; i < keypoints.size(); i++)
        {
            points.push_back(keypoints[i].pt);
        }
    }*/
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

    void Match(Mat img_1, Mat img_2, vector<Point2f> & points0, vector<Point2f> & points1)
    {
        BFMatcher matcher(NORM_HAMMING);
        Ptr<ORB> orb = ORB::create(1000, 2, 4, 5, 0, 2, ORB::HARRIS_SCORE, 5);
        vector<Mat> descriptors;
        vector<KeyPoint> keypoints_1, keypoints_2;
        Mat descriptors_1, descriptors_2;
        //orb->detect(img_1, keypoints_1);
        //orb->detect(img_2, keypoints_2);
        for (size_t i = 0; i < points0.size(); i++) {
            keypoints_1.push_back(cv::KeyPoint(points0[i], 1.f));
        }
        for (size_t i = 0; i < points1.size(); i++) {
            keypoints_2.push_back(cv::KeyPoint(points1[i], 1.f));
        }
        orb->compute(img_1, keypoints_1, descriptors_1);
        orb->compute(img_2, keypoints_2, descriptors_2);
        descriptors.push_back(descriptors_1);
        matcher.add(descriptors);
        vector<vector<DMatch>> matches, goodmatches;
        DMatch bestMatch, betterMatch;
        vector<DMatch> bestMatches;
        matcher.knnMatch(descriptors_2, matches, 2);
        int n = 0;
        Point p1, p2;
        points0.clear();
        points1.clear();
        for (int i = 0; i<(int)matches.size(); i++)
        {
            bestMatch = matches[i][0];
            betterMatch = matches[i][1];
            p1 = keypoints_1[bestMatch.trainIdx].pt;
            p2 = keypoints_2[bestMatch.queryIdx].pt;
            double distance = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
            float distanceRatio = bestMatch.distance / betterMatch.distance;

            if (distanceRatio< 0.8 && distance<30)
            {
                bestMatches.push_back(bestMatch);
                points0.push_back(p1);
                points1.push_back(p2);
                //line(img_2, p1, p2, Scalar(0, 0, 255), 1, 8, 0);
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


#endif //LOOP_CLOSURE_TRACKING_H
