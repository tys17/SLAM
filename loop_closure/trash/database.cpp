//
// Created by root on 12/3/17.
//

#include "database.h"

bool database::addKeyframe(keyframe &frame) {
    frame.index = keyframes.size();
    DBoWdatabase.add(frame.descriptors);
    return true;
}

QueryResults database::queryKNN(keyframe &frame, int k) {
    QueryResults res;
    DBoWdatabase.query(frame.descriptors, res, k);
    return res;
}

int database::detectLoop(keyframe &frame) {
    QueryResults res = queryKNN(frame, neighborNum);
    if (res.size() <= 0)
        return -1;
    else{
        int currentID = keyframes.size();
        int minimalID = currentID + 1;
        keyframe minimalFrame;
        for (size_t i = 0; i < res.size(); i++){
            if (res[i].Score >= thres && currentID - res[i].Id >= minimumIDInterval &&
                    frame.timestamp - keyframes[res[i].Id].timestamp >= minimumTimeInterval){
                if (BAOnce && !keyframes[res[i].Id].BAed){
                    if (res[i].Id < minimalID){
                        minimalID = res[i].Id;
                        minimalFrame = keyframes[res[i].Id];
                    }
                }
                else{
                    if (res[i].Id < minimalID){
                        minimalID = res[i].Id;
                        minimalFrame = keyframes[res[i].Id];
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