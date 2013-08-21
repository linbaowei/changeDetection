/*
 *   Copyright (c) 2012 <Baowei Lin> <lin-bao-wei@hotmail.com>
 * 
 *   Permission is hereby granted, free of charge, to any person
 *   obtaining a copy of this software and associated documentation
 *   files (the "Software"), to deal in the Software without
 *   restriction, including without limitation the rights to use,
 *   copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the
 *   Software is furnished to do so, subject to the following
 *   conditions:
 * 
 *   The above copyright notice and this permission notice shall be
 *   included in all copies or substantial portions of the Software.
 * 
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *   OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *   OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef SIFTFEATUREDETECTION_H
#define SIFTFEATUREDETECTION_H

#include <iostream>
#include "GL/glew.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/GlobalUtil.h"
#include "cv.h"
#include "highgui.h"
#include <vector>
using namespace std;
using namespace cv;

class SIFTFeatureDetection
{
  
public:
    SiftGPU sift;
    SIFTFeatureDetection();
    int CallSiftGPU (Mat colorImage, int &FeatureNum, vector<SiftGPU::SiftKeypoint> &keypoint,vector<unsigned int> &dDescriptors);
    ~SIFTFeatureDetection();
};

#endif // SIFTFEATUREDETECTION_H







