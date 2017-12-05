#ifndef LOOP_CLOSURE_DATABASE_H
#define LOOP_CLOSURE_DATABASE_H
#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "DBoW3/DBoW3.h"

#include <iostream>
#include <fstream>
#include<algorithm>
#include <sstream>
#include <stdio.h>
//#include <opencv.hpp>
//#include <core/core.hpp>
//#include <highgui/highgui.hpp>
//#include <features2d/features2d.hpp>
//#include <windows.h>
#include<math.h>
//#include<io.h>
#include <string>
#include <fstream>
#include <sstream>
//#include <direct.h>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

//#include <Windows.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/xfeatures2d.hpp"
//#include <windows.h>
//#include <tchar.h>
#include <stdio.h>
//#include <strsafe.h>
#include<vector>
#include<unordered_map>

using namespace std;
using namespace cv;


class KeyFrame
{
public:
    cv::Mat R;
    cv::Mat t;
    bool BAed = false;
    cv::Mat descriptor;
    //std::unordered_map<int, cv::Mat> feat_2_descrip;    //feature_id to descriptor
    std::vector <int> feature_id;
    int frame_id;
    std::vector<int> loop_closed_with;
    double time_stamp = 0;
    KeyFrame() { R.create(3, 3, CV_64FC1); t.create(3, 1, CV_64FC1); }
    KeyFrame(std::vector <int>  k, Mat d, int frame_id1, cv::Mat R1, cv::Mat t1) :frame_id(frame_id1) {
        R = R1.clone();
        t = t1.clone();
        feature_id = k; descriptor = d.clone();
    }
};

class Feature
{
public:
    int feature_id;
    cv::Point3f coord_3D;
    std::unordered_map <int, cv::Point2f> frame_xy;   //The coordinate in frames
    Feature(int feature_id1, cv::Point3f coord_3D1) :feature_id(feature_id1), coord_3D(coord_3D1) {}
};

class Database {
public:
    DBoW3::Vocabulary voc;
    DBoW3::Database DBoWdatabase;
    std::vector <KeyFrame> frame_list;;
    std::vector<Feature> feature_list;
    // loop closure parameters
    // to be modified
    int neighborNum = 20;
    double thres = 0.9;
    int minimumTimeInterval = 100;
    int minimumIDInterval = 10;
    bool BAOnce = true;

    void AddFrame(KeyFrame add_frame)
    {
        frame_list.push_back(add_frame);
    }
    void AddFeature(Feature add_feature)
    {
        feature_list.push_back(add_feature);
    }
    Database(DBoW3::Vocabulary v) { voc = v; createDBoWDatabase();}
    bool createDBoWDatabase() { DBoW3::Database tmp(voc, false, 0); DBoWdatabase = tmp; return true; }
    bool addKeyframe(KeyFrame frame);
    DBoW3::QueryResults queryKNN(KeyFrame& frame, int k);
    int detectLoop(KeyFrame& frame);
};
//class Keyframe_DB
//{
//public:
//	std::vector <KeyFrame> frame_list;
//	void AddFrame(KeyFrame add_frame)
//	{
//		frame_list.push_back(add_frame);
//	}
//};
//class Feature_DB
//{
//public:
//	std::vector<Feature*> feature_id_list;
//	void AddFeature(Feature* add_feature)
//	{
//		feature_id_list.push_back(add_feature);
//	}
//};
#endif