//
// Created by root on 12/3/17.
//

#ifndef LOOP_CLOSURE_FILEIO_H
#define LOOP_CLOSURE_FILEIO_H

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
using namespace std;

class fileIO {
public:
    vector<string> loadImages(string folderDir);

};


#endif //LOOP_CLOSURE_FILEIO_H
