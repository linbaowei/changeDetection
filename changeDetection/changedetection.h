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

#ifndef CHANGEDETECTION_H
#define CHANGEDETECTION_H

#include <iostream>
#include <fstream>
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include "boost/program_options.hpp"

#include "options.h"
#include "readFeatures.h"

using namespace std;


#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "cv.h"
#include "highgui.h"
using namespace cv;

#include "readFeatures.h"
#include "featurematching.h"
#include "fileoperation.h"
#include "pmat_ransac.h"

struct correspondingTestToTrain{
  int traingNo;
  double dis;
};

struct correspondingTestToTrains{    
  int testNo;
  vector<correspondingTestToTrain> corresVec;
};


class changedetection
{

public:
  SiftGPU  sift;	
  FeatureMatching featurematching;
  vector< vector<Point2d> > keypoints_trainingimages;
  vector<  vector<unsigned int> > descriptors_trainingimages;
  vector<string> trainingImagenames;
  vector<string> filenames;
  vector<string> testimagenames;
  int topNearestImagesNum;
  string trainingImagesKeysPath;
  string testImagesPath;
  
  
  changedetection ();
  
  int drawmatching(string trainingimg,
		   string testimg,  
		   vector<Point2d> keypoints_testimages, 
		   vector<correspondingPair> corres);
  
  Mat matchingStart( vector<Point2d> keypoints_testimages,
		   vector<unsigned int> descriptors_testimages, 
		   vector<Point3d> keypoints_3Dpoints, 
		   vector<unsigned int> descriptors_3Dpoints,
		   mats  matrices,
		   string imageForShow,
		   int flag);
  
  Mat matchingStart1( vector<Point2d> keypoints_testimages,
				    vector<unsigned int> descriptors_testimages, 
				    vector<Point3d> keypoints_3Dpoints, 
				    vector<unsigned int> descriptors_3Dpoints,
				    mats  matrices,
				    Mat intrinsic,
				    Mat distCoeffs,
				    string imageForShow,
				    int flag);
  
  Mat matchingStart2( vector<Point2d> keypoints_testimages,
		   vector<unsigned int> descriptors_testimages, 
		   vector<Point3d> keypoints_3Dpoints, 
		   vector<unsigned int> descriptors_3Dpoints,
		   mats  matrices,
		   string imageForShow,
		   int flag);
  
  string distanceofRotationAndTranslation(Mat orignalpose, 
					vector<Mat> cameraposes, 
					vector<string> trainingimages);
  
  
  void changeImageSize(const string trainingimagePath, const string resizedPath);
  string experiment();
  void draw2Dkeypoints(vector<correspondingPair> corres,
				vector<Point2d> keypoints_testimages,
				string  imageForShow );
  
  Mat RaboutX(const double& theta);
  Mat RaboutY(const double& theta);
  Mat RaboutZ(const double& theta);
  virtual ~ changedetection ();
};

#endif // CHANGEDETECTION_H
