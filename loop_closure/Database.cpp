//
// Created by root on 12/3/17.
//

#include "Database.h"

bool Database::addKeyframe(KeyFrame frame) {
    //frame.frame_id = frame_list.size();
    KeyFrame tmp(frame.feature_id, frame.descriptor, frame.frame_id, frame.R, frame.t);
    DBoWdatabase.add(frame.descriptor);
    frame_list.push_back(tmp);
    //frame_list.push_back(frame);
    return true;
}


DBoW3::QueryResults Database::queryKNN(KeyFrame &frame, int k) {
    DBoW3::QueryResults res;
    DBoWdatabase.query(frame.descriptor, res, k);
    /*if (frame_list.size() == 593) {
        cout << res << endl;
        for (int j = 0; j < res.size(); j++){
            //cout << res[j].Score << endl;
            //cout << res[j].Score << " " << res[j].expectedChiScore << " " << res[j].chiScore << " " << res[j].bhatScore << endl;
        }

    }*/
    return res;
}

int Database::detectLoop(KeyFrame &frame) {
    DBoW3::QueryResults res = queryKNN(frame, neighborNum);
    if (res.size() <= 0)
        return -1;
    else{
        int currentID = frame_list.size();
        int minimalID = currentID + 1;
        KeyFrame minimalFrame;
        for (size_t i = 0; i < res.size(); i++){
            if (currentID - res[i].Id >= minimumIDInterval){
                /*if (frame_list.size() == 593)
                    cout << "test" << endl;*/
                vector<DMatch> goodMatches = computeGoodMatches(frame, frame_list[res[i].Id]);
                if (goodMatches.size() >= minimumGoodMatches) {
                    if (res[i].Id < minimalID) {
                        minimalID = res[i].Id;
                        minimalFrame = frame_list[res[i].Id];
                    }
                }
            }
        }
        if (minimalID == currentID + 1)
            return -1;
        else
            return minimalID;
    }
}

vector<DMatch> Database::computeGoodMatches(KeyFrame &frame1, KeyFrame &frame2) {
    vector<DMatch> matches;
    frame1.descriptor.convertTo(frame1.descriptor, CV_32F);
    frame2.descriptor.convertTo(frame2.descriptor, CV_32F);
    matcher.match(frame1.descriptor, frame2.descriptor, matches);
    double min_dist = 100000;
    for (size_t i = 0; i < matches.size(); i++){
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
    }
    vector<DMatch> res;
    for (size_t i = 0; i < matches.size(); i++){
        double dist = matches[i].distance;
        if (dist < goodMatchDist)
            res.push_back(matches[i]);
    }
    /*if (frame_list.size() == 593) {
        cout << min_dist << endl;
        for (int j = 0; j < matches.size(); j++) {
            cout << matches[j].distance << endl;
            //cout << res[j].Score << endl;
            //cout << res[j].Score << " " << res[j].expectedChiScore << " " << res[j].chiScore << " " << res[j].bhatScore << endl;
        }
        cout << res.size() << endl;
    }*/
    //cout << min_dist << endl;
    //cout << res.size() << endl;
    return res;
}