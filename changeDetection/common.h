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


#include <iostream>
using namespace std;

#include <fstream>
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include "boost/program_options.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "cv.h"
#include "highgui.h"
using namespace cv;

#include "readFeatures.h"
#include "featurematching.h"
#include "fileoperation.h"
#include "changedetection.h"
#include "poseestimation.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/GlobalUtil.h"
#include "options.h"



/**
 *  codes before review commonds(before 20120628)
 */
options option;
void methodsift()
{
	string allInOne3Dpoints;
	string plyfilePath;
	string all3DpointsBefroeThresholding; 
	changedetection detection;
	
	
	allInOne3Dpoints = option.ThreeDKeypointsFile;
	plyfilePath = option.plyFileof3DpointCloudFile_train;  
	detection.trainingImagesKeysPath = option.trainingImagesKeysPath;
	detection.testImagesPath = option.testImagesPath;  
	//detection.experiment();
	//detection.changeImageSize(option.trainingImagesKeysPath, option.resizedImagesPath); 
	
	vector<Point3d> keypoints_3Dpoints;
	vector<unsigned int> descriptors_3Dpoints;
	
	FeatureMatching featurematch;
	fileOperation fileoperation; 
	
	mats matrices;  
	detection.trainingImagenames = fileoperation.getFilenames(detection.trainingImagesKeysPath, "*.key");
	for(int i = 0; i < detection.trainingImagenames.size(); i++)
	{
		vector<Point2d> keypoints_trainingimage;
		vector<unsigned int> descriptors_trainingimage;
		readFeaturestoVec(detection.trainingImagesKeysPath+detection.trainingImagenames.at(i),
					keypoints_trainingimage,
					descriptors_trainingimage);
		detection.keypoints_trainingimages.push_back(keypoints_trainingimage);
		detection.descriptors_trainingimages.push_back(descriptors_trainingimage);
		cout << "reading " << detection.trainingImagenames.at(i) << endl; 
	} 
	
	read3DFeaturestoVec(allInOne3Dpoints, 
				keypoints_3Dpoints,
				descriptors_3Dpoints);
	
	matrices = featurematch.read3Dpoints(plyfilePath);	
	detection.filenames = fileoperation.getFilenames(detection.testImagesPath, "*.key");
	
	for(int i = 0; i < detection.filenames.size(); i++)
	{
		//Mat imageforshow = imread(testImagesPath + filenames.at(i).substr(0, filenames.at(i).length()-4)+".jpg");
		string imageforshow = detection.testImagesPath + detection.filenames.at(i).substr(0, detection.filenames.at(i).length()-4)+".jpg";
		vector<Point2d> keypoints_testimages;
		vector<unsigned int> descriptors_testimages;
		readFeaturestoVec(detection.testImagesPath+detection.filenames.at(i),
					keypoints_testimages,
					descriptors_testimages);
		cout << "reading " << detection.filenames.at(i) << endl; 
		detection.matchingStart(keypoints_testimages, 
					descriptors_testimages, 
					keypoints_3Dpoints, 
					descriptors_3Dpoints, 
					matrices, 
					imageforshow,i);
	}   
}

int siftExtraction(string path, Mat videoframe)
{ 
	SiftGPU sift;
	string savekeyfile =path + ".key";
	cout << savekeyfile << endl;
	if(videoframe.empty())
	{	
		cerr << "sift exit." << endl;
		return -1;	    
	}
	unsigned char *data=videoframe.ptr();   	
	
	char * argg[] ={ "-fo", "-1", "-v", "0"};//,"-d","10"
	//-fo -1,  starting from -1 octave
	//-v 1,  only print out # feature and overall time
	sift.ParseParam(4, argg);    
	int support = sift.CreateContextGL();	
	if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED)	
		exit(1);
	sift.RunSIFT(videoframe.size().width, 
			videoframe.size().height, 
			data, GL_RGB, 
			GL_UNSIGNED_BYTE);
	sift.SaveSIFT(savekeyfile.c_str());
	int FeatureNum=sift.GetFeatureNum();
	float* descriptors = new float[FeatureNum];
	SiftGPU::SiftKeypoint* keys = new SiftGPU::SiftKeypoint[FeatureNum];    
	//extract the features
	//sift.GetFeatureVector(keys, descriptors);
	sift.SaveSIFT(savekeyfile.c_str());
	cerr << "key file of " << savekeyfile << "is generated currectly." << endl; 
	return 1;
}
	
void methodopencvPoseEstimation(Mat intrinsic, Mat distCoeffs)
{
	string allInOne3Dpoints;
	string plyfilePath;
	string all3DpointsBefroeThresholding; 
	changedetection detection;
	
	
	allInOne3Dpoints = option.ThreeDKeypointsFile;
	plyfilePath = option.plyFileof3DpointCloudFile_train;  
	detection.trainingImagesKeysPath = option.trainingImagesKeysPath;
	detection.testImagesPath = option.testImagesPath;  
	detection.topNearestImagesNum = option.topNearestImagesNum;
	//detection.experiment();
	//detection.changeImageSize(option.trainingImagesKeysPath, option.resizedImagesPath); 
	
	vector<Point3d> keypoints_3Dpoints;
	vector<unsigned int> descriptors_3Dpoints;
	
	FeatureMatching featurematch;
	fileOperation fileoperation; 
	
	mats matrices;  
	detection.trainingImagenames = fileoperation.getFilenames(detection.trainingImagesKeysPath, "*.key");
	cout << "reading :"<< endl;
	for(int i = 0; i < detection.trainingImagenames.size(); i++)
	{
		vector<Point2d> keypoints_trainingimage;
		vector<unsigned int> descriptors_trainingimage;
		readFeaturestoVec(detection.trainingImagesKeysPath+detection.trainingImagenames.at(i),
					keypoints_trainingimage,
		descriptors_trainingimage);
		detection.keypoints_trainingimages.push_back(keypoints_trainingimage);
		detection.descriptors_trainingimages.push_back(descriptors_trainingimage);
		cout <<detection.trainingImagenames.at(i) << "\t"; 
	} 
	cout << endl;
	
	read3DFeaturestoVec(allInOne3Dpoints, 
				keypoints_3Dpoints,
				descriptors_3Dpoints);
	
	matrices = featurematch.read3Dpoints(plyfilePath);	
	detection.filenames = fileoperation.getFilenames(detection.testImagesPath, "*.key");
	cout << "\n\n\nThere are " << detection.filenames.size() << " *.key files exist in " << detection.testImagesPath << endl;
	vector<string> imagenamesjpg = fileoperation.getFilenames(detection.testImagesPath, "*.jpg");
	vector<string> imagenamesJPG = fileoperation.getFilenames(detection.testImagesPath, "*.JPG");
	vector<string> imagenames = imagenamesjpg;
	for(int sizetmp = 0; sizetmp < imagenamesJPG.size(); ++sizetmp)
		imagenames.push_back(imagenamesJPG.at(sizetmp));
	detection.testimagenames = imagenames;
	if(imagenames.size() > 1)
	{
		cout << "\n\n\nPlease just keep ONE test image in " << detection.testImagesPath << endl;
		cout << "\n\n\nSystem exit!" << endl;
		exit(1);
	}
	if(detection.filenames.size() < imagenames.size())
	{
		cout << "there is no key file for  query images. Now generating..." << endl;	
		for(int a = 0; a < imagenames.size(); a ++)
		{
			Mat image = imread(detection.testImagesPath+imagenames.at(a));	   
			string tmpstr = detection.testImagesPath+imagenames.at(a);
			siftExtraction(tmpstr.substr(0, tmpstr.length()-4), image);	    
		}
		detection.filenames.clear();
		detection.filenames = fileoperation.getFilenames(detection.testImagesPath, "*.key");
		cout << detection.filenames.at(0) << endl;
	}
	for(int i = 0; i < detection.filenames.size(); i++)
	{
		//Mat imageforshow = imread(testImagesPath + filenames.at(i).substr(0, filenames.at(i).length()-4)+".jpg");
		//string imageforshow = detection.testImagesPath + detection.filenames.at(i).substr(0, detection.filenames.at(i).length()-4)+".jpg";
		string imageforshow = detection.testImagesPath + imagenames.at(i);
		vector<Point2d> keypoints_testimages;
		vector<unsigned int> descriptors_testimages;
		readFeaturestoVec(detection.testImagesPath+detection.filenames.at(i),
					keypoints_testimages,
		descriptors_testimages);
		cout << "reading " << detection.filenames.at(i) << endl; 
		detection.matchingStart1(keypoints_testimages, 
					descriptors_testimages, 
					keypoints_3Dpoints, 
					descriptors_3Dpoints,
					matrices,
					intrinsic, 
					distCoeffs,
					imageforshow,i);
	}   
}