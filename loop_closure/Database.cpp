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
    if (frame_list.size() == 593) {
        cout << res << endl;
        for (int j = 0; j < res.size(); j++){
            //cout << res[j].Score << endl;
            //cout << res[j].Score << " " << res[j].expectedChiScore << " " << res[j].chiScore << " " << res[j].bhatScore << endl;
        }

    }
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
            if (res[i].Score >= thres){
                if (res[i].Id < minimalID){
                    minimalID = res[i].Id;
                    minimalFrame = frame_list[res[i].Id];
                }
            }
        }
        if (minimalID == currentID + 1)
            return -1;
        else
            return minimalID;
    }
}