//
// Created by root on 12/3/17.
//

#ifndef LOOP_CLOSURE_DATABASE_H
#define LOOP_CLOSURE_DATABASE_H
#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "DBoW3/DBoW3.h"
using namespace std;
using namespace cv;
using namespace DBoW3;

template <class descriptor>
class feature {
public:
    Point2f coordinate;    //xy coordinate
    descriptor data;    //actual data of descriptor
};

class keyframe {
public:
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat R;
    Mat T;
    bool BAed = false;
    int index;
    double timestamp = 0;
    keyframe(){R.create(3, 3, CV_64FC1); T.create(3, 1, CV_64FC1);}
    keyframe(vector<KeyPoint> k, Mat d){keypoints = k; descriptors = d;R.create(3, 3, CV_64FC1); T.create(3, 1, CV_64FC1);}
};


class database {
public:
    Vocabulary voc;
    DBoW3::Database DBoWdatabase;
    vector<keyframe> keyframes;

    // loop closure parameters
    // to be modified
    int neighborNum = 20;
    double thres = 0.7;
    int minimumTimeInterval = 100;
    int minimumIDInterval = 10;
    bool BAOnce = true;

    database(Vocabulary v) {voc = v;}
    bool createDBoWDatabase() {DBoW3::Database tmp(voc, false, 0); DBoWdatabase = tmp; return true;}
    bool addKeyframe(keyframe& frame);
    QueryResults queryKNN(keyframe& frame, int k);
    int detectLoop(keyframe& frame);
};


#endif //LOOP_CLOSURE_DATABASE_H
