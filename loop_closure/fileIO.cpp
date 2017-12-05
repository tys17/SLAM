//
// Created by root on 12/3/17.
//

#include "fileIO.h"

vector<string> fileIO::loadImages(string folderDir) {
    vector<string> res;
    if (folderDir.length() > 0 && folderDir[folderDir.length() - 1] != '/')
        folderDir = folderDir + "/";
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(folderDir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << folderDir << endl;
        return res;
    }

    while ((dirp = readdir(dp)) != NULL) {
        string filename = string(dirp->d_name);
        if (filename != "." && filename != "..")
            res.push_back(folderDir + string(dirp->d_name));
    }
    closedir(dp);
    sort(res.begin(), res.end());
    return res;
}