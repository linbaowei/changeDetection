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

#ifndef __READFEATURES__
#define __READFEATURES__

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <string>
#include "options.h"
using namespace std;

#include "cv.h"
#include "highgui.h"
using namespace cv;

#include "featurematching.h"

void readAllFeatureFiles(const string inputFilename,
			 vector<string> &allFilenames);

void readVecData(const options Option,
			  vector<double> &vectortrain,
			  vector< vector<double> > &trainingVectorData,
			  vector<double> &vectorquery,
			  vector< vector<double> > &testVectorData);

void vector_distance(const vector<double> &vectorData1,
			      const vector<double> &vectorData2,
			      double &dis);


void readFeaturestoMat(const string inputFilename,
				    Mat &descriptor);

void readFeaturestoVec(const string inputFilename,
				    vector<Point2d> &keypointsCoor,
				    vector<unsigned int> &descriptor);

void read3DFeaturestoVec(const string inputFilename,
				    vector<Point3d> &keypointsCoor,
				    vector<unsigned int> &descriptor);

void readMatching3Dto2D(string correspondings3D2D,vector<matchedPoints> &matchedPoints3Dto2D);
#endif
