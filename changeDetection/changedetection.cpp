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


#include "changedetection.h"
#include "options.h"
#include "readFeatures.h"
#include "featurematching.h"
#include "fileoperation.h"
#include "pmat_ransac.h"
#include "poseestimation.h"

changedetection::changedetection ()
{
	
}

changedetection::~changedetection ()
{
	
}

static int cmp( pair<string, vector<correspondingPair> >  i, pair<string, vector<correspondingPair> > j){
	if( i.second.size() < j.second.size())
		return 0;
	else
		return 1;
}

int changedetection::drawmatching(string trainingimg, string testimg,  vector<Point2d> keypoints_testimages, vector<correspondingPair> corres)
{
// 	const char* imagename1=trainingimg.c_str();
// 	const char* imagename2=testimg.c_str();   
// 	IplImage* mergetImage;
// 	IplImage* image1 = cvLoadImage(imagename1, 1);
// 	IplImage* image2 = cvLoadImage(imagename2, 1);
// 	cout << "draw matches." << endl;
// 	CvSize size;
// 	size.width = image1->width + image2->width + 10;
// 	size.height = (image1->height > image2->height) ? image1->height : image2->height;
// 	mergetImage = cvCreateImage(size,image1->depth,image1->nChannels);
// 	CvRect rect = cvRect(0,0,image1->width,image1->height);
// 	cvSetImageROI(mergetImage,rect);
// 	cvRepeat(image1,mergetImage);
// 	cvResetImageROI(mergetImage);
// 	rect = cvRect(image1->width + 10,0,image2->width,image2->height);
// 	cvSetImageROI(mergetImage,rect);
// 	cvRepeat(image2,mergetImage);
// 	cvResetImageROI(mergetImage);
// 	for(int i = 0; i < corres.size() ; i++) {
// 		cvLine(mergetImage,
// 		       cvPoint(keypoints_testimages[corres.at(i).indexInTest].x + image1->width + 10,
// 			       keypoints_testimages[corres.at(i).indexInTest].y),
// 		       cvPoint(keypoints_trainingimages.at(0).at(corres.at(i).indexInTraining).x,
// 			       keypoints_trainingimages.at(0).at(corres.at(i).indexInTraining).y),
// 		       CV_RGB(0,255,255),
// 		       1,
// 	 0);
// 		i+=10;
// 	}
// 	//namedWindow("vertical",CV_WINDOW_AUTOSIZE);
// 	//imshow("vertical", mergetImage);
// 	Mat show(mergetImage);
    
    
//     featurematching;    
//     static int count=0;
//     imwrite("vertical"+featurematching.IntToString(count)+".jpg", show); 
//     count ++;
    //waitKey();
}






Mat changedetection::matchingStart( vector<Point2d> keypoints_testimages,
				    vector<unsigned int> descriptors_testimages, 
				    vector<Point3d> keypoints_3Dpoints, 
				    vector<unsigned int> descriptors_3Dpoints,
				    mats  matrices,
				    string imageForShow,
				    int flag)
{
	
	//matching descriptors
	vector<correspondingPair> correspondings_origianl;  //两个descriptor文件的比较对应序列
	vector<correspondingPair> correspondings;
	
	
	TickMeter tm;
	tm.start();	    
	featurematching.MatchingStart(128,
				      0.0,
			       descriptors_testimages,     // features of each training image
			       descriptors_3Dpoints,   // features of test image
			       correspondings_origianl);  	    
	
	tm.stop();
	double buildTime = tm.getTimeMilli();
	cout << "Matching time: " << buildTime << " ms" << endl;
	
	vector<matchedPoints> matchedPointsfianl;
	
	//correspondings = featurematching.getCorrespondingsAfterErasingRepeats(correspondings_origianl);
	
	correspondings = correspondings_origianl;
	Mat matrix3Dpoints = Mat(4, correspondings.size(), CV_64F);
	for(int i =0; i < correspondings.size(); i ++)
	{
		
		matchedPoints onepair;
		onepair.x3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).x;
		onepair.y3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).y;
		onepair.z3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).z;
		onepair.x2d = keypoints_testimages.at(correspondings.at(i).indexInTraining).x;
		onepair.y2d = keypoints_testimages.at(correspondings.at(i).indexInTraining).y;
		matchedPointsfianl.push_back(onepair);
		matrix3Dpoints.at<double>(0, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).x;
		matrix3Dpoints.at<double>(1, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).y;
		matrix3Dpoints.at<double>(2, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).z;
		matrix3Dpoints.at<double>(3, i) = 1;
		
		//      cout << i<<"\t"<<onepair.x3d << "\t" 
		//     	       << onepair.y3d << "\t" 
		//     	       <<onepair.z3d<< "\t" 
		//     	       << onepair.x2d << "\t" 
		//     	       << onepair.y2d<< endl;
	}
	
	TickMeter tm1;
	tm1.start();	    
	/*******************projection matrix estimation BEGIN*********************/
	Mat MatrixP = pmat_ransac(matchedPointsfianl);
	cout << MatrixP << endl;
	/****to get intrinsic matrix****/
	Mat matrixk=Mat(3,3,CV_64FC1);	//intrinsic matrix
	Mat R_trainingimg=Mat(3,3,CV_64FC1);  
	Mat T_trainingimg4x1=Mat(4,1,CV_64FC1);
	Mat T_trainingimg3x1=Mat(3,1,CV_64FC1);
	cv::decomposeProjectionMatrix(MatrixP,
				      matrixk,
			       R_trainingimg,
			       T_trainingimg4x1);   
	T_trainingimg3x1=matrixk.inv()*MatrixP.col(3);
	T_trainingimg3x1.at<double>(0, 0) += 0;
	T_trainingimg3x1.at<double>(0, 1) -= 0;
	T_trainingimg3x1.at<double>(0, 2) -= 0;
	
	
	double rotationangle =10 * 3.1415 / 180;
	Mat rotation = RaboutY(rotationangle) * R_trainingimg;
	
	Mat projectionMatirx = Mat(3,4,CV_64FC1);
	for(int i =0; i < 3; i ++)
	{
		projectionMatirx.at<double>(0, i) = rotation.at<double>(0, i);
		projectionMatirx.at<double>(1, i) = rotation.at<double>(1, i);
		projectionMatirx.at<double>(2, i) = rotation.at<double>(2, i);
	}
	projectionMatirx.at<double>(0, 3) = T_trainingimg3x1.at<double>(0, 0);
	projectionMatirx.at<double>(1, 3) = T_trainingimg3x1.at<double>(1, 0);
	projectionMatirx.at<double>(2, 3) = T_trainingimg3x1.at<double>(2, 0);
	
	/*******************projection matrix estimation END*********************/
	MatrixP = matrixk*projectionMatirx;
	cout << MatrixP << endl;
	tm1.stop();
	double buildTime1 = tm1.getTimeMilli();
	cout << "RANSAC time: " << buildTime1 << " ms" << endl;  
	
	if(MatrixP.at<double>(2,3)==0)
		return MatrixP;
	else
	{    	
	    Mat tmpmat = imread(imageForShow);
	    
	    Mat mattmp = featurematching.project(matrices.matrix3d, MatrixP,  tmpmat);
	    // 
	    //    cout  << "camera pose is :\n"<<MatrixP << endl;
	    imwrite("result_"+ filenames.at(flag)+".jpg", mattmp);
	    
	    
	    vector<correspondingPair> corres;
	    string flename;
	    vector< pair<string, vector<correspondingPair> > > allcorres;
	    
	    for(int i =0; i < descriptors_trainingimages.size(); i++)
	    { 
		    vector<correspondingPair> correstmp;
		    TickMeter tm;
		    tm.start();
		    featurematching.MatchingStart2(128,
						    0.0,
						    descriptors_trainingimages.at(i),     // features of each training image
						    descriptors_testimages,   // features of test image
						    correstmp);  
		    descriptors_trainingimages.at(i).clear();
		    tm.stop();
		    double buildTime = tm.getTimeMilli();
		    cout << "total matching time: " << buildTime << " ms" << endl;
		    cout << filenames.at(flag) << endl; 
		    cout << trainingImagenames.at(i) << endl;
		    cout << correstmp.size() << endl;    
		    
		    
		    
		    draw2Dkeypoints(correstmp, keypoints_testimages, imageForShow);
		    string trainimgtmp = trainingImagesKeysPath+trainingImagenames.at(i).substr(0, trainingImagenames.at(i).length()-4);
		    stringstream ss(trainimgtmp);
		    string sub_str;
		    while(getline(ss,sub_str,'/')) //以|为间隔分割test的内容
				continue;
		    string inputimagepath = trainimgtmp.substr(0, trainimgtmp.find_last_of(sub_str) - sub_str.length() + 1);
		    string filename = trainimgtmp.substr(trainimgtmp.find_last_of(sub_str)  - sub_str.length() + 1, trainimgtmp.length() -1);
		    
		    string zero = "0";
		    if(filename.length() < 8)
		    {		
			for(int a = 1; a < 8 - filename.length(); a ++)
			    zero += "0";
		    }
		    filename = zero + filename;						
		    string trainimg = inputimagepath + filename +".jpg";
		    string testimg = testImagesPath + filenames.at(flag).substr(0,filenames.at(flag).length()-4)+".jpg";
		    cout << trainimg << "\n" << testimg << endl;
		    //drawmatching(trainimg, testimg, keypoints_testimages, correstmp);
		    pair<string, vector<correspondingPair> > pa;
		    pa.first = trainingImagenames.at(i);
		    pa.second = correstmp;
		    allcorres.push_back(pa);
	    }
	    for(int i = 0; i < allcorres.size(); i++)
	    {
		    cout  <<allcorres.at(i).second.size()<< endl;
	    }
	    
	    sort(allcorres.begin(), allcorres.end(), cmp);
	    Mat matforshow;
	    vector<int> distanceofpixel;
	    distanceofpixel.push_back(0);
	    distanceofpixel.push_back(5);
	    distanceofpixel.push_back(10);
	    distanceofpixel.push_back(20);
	    distanceofpixel.push_back(30);
	    distanceofpixel.push_back(50);
	    distanceofpixel.push_back(70);
	    distanceofpixel.push_back(90);
	    
	    for(int p = 0; p < 7; p++)
	    {
		    cout  << allcorres.at(p).first << "\t" << allcorres.at(p).second.size() <<"---------------------"<< endl;
		    for(int a = 0; a < distanceofpixel.size(); a++)
		    {
			    matforshow = featurematching.projection(matrices,
								    MatrixP, 
								    allcorres.at(p).second, 
								    keypoints_testimages, 
								    imageForShow,  
								    filenames.at(flag)+allcorres.at(p).first, 
								    distanceofpixel.at(a));	   
		    }
	    }
	    //Mat matforshow;
	    return matforshow;
	}
}

Mat changedetection::matchingStart1( vector<Point2d> keypoints_testimages,
				    vector<unsigned int> descriptors_testimages, 
				    vector<Point3d> keypoints_3Dpoints, 
				    vector<unsigned int> descriptors_3Dpoints,
				    mats  matrices,
				    Mat intrinsic,
				    Mat distCoeffs,
				    string imageForShow,
				    int flag)
{
	
	//matching descriptors
	vector<correspondingPair> correspondings_origianl;  //两个descriptor文件的比较对应序列
	vector<correspondingPair> correspondings;
	
	
	TickMeter tm;
	tm.start();	    
	featurematching.MatchingStart(128,
				      0.0,
			       descriptors_testimages,     // features of each training image
			       descriptors_3Dpoints,   // features of test image
			       correspondings_origianl);  	    
	
	tm.stop();
	double buildTime = tm.getTimeMilli();
	cout << "Matching time: " << buildTime << " ms" << endl;
	
	vector<matchedPoints> matchedPointsfianl;
	
	//correspondings = featurematching.getCorrespondingsAfterErasingRepeats(correspondings_origianl);
	
	correspondings = correspondings_origianl;
	Mat matrix3Dpoints = Mat(4, correspondings.size(), CV_64F);
	for(int i =0; i < correspondings.size(); i ++)
	{
		
		matchedPoints onepair;
		onepair.x3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).x;
		onepair.y3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).y;
		onepair.z3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).z;
		onepair.x2d = keypoints_testimages.at(correspondings.at(i).indexInTraining).x;
		onepair.y2d = keypoints_testimages.at(correspondings.at(i).indexInTraining).y;
		matchedPointsfianl.push_back(onepair);
		matrix3Dpoints.at<double>(0, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).x;
		matrix3Dpoints.at<double>(1, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).y;
		matrix3Dpoints.at<double>(2, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).z;
		matrix3Dpoints.at<double>(3, i) = 1;
		
		//      cout << i<<"\t"<<onepair.x3d << "\t" 
		//     	       << onepair.y3d << "\t" 
		//     	       <<onepair.z3d<< "\t" 
		//     	       << onepair.x2d << "\t" 
		//     	       << onepair.y2d<< endl;
	}
	
	TickMeter tm1;
	tm1.start();	    
	/*******************projection matrix estimation BEGIN*********************/
	poseEstimation poseestimation;
	Mat rvec, tvec;
	Mat MatrixP = poseestimation.findPose(matchedPointsfianl, intrinsic, distCoeffs, rvec, tvec);
	cout << MatrixP << endl;
	/****to get intrinsic matrix****/
	Mat matrixk=Mat(3,3,CV_64FC1);	//intrinsic matrix
	Mat R_trainingimg=Mat(3,3,CV_64FC1);  
	Mat T_trainingimg4x1=Mat(4,1,CV_64FC1);
	Mat T_trainingimg3x1=Mat(3,1,CV_64FC1);
	cv::decomposeProjectionMatrix(MatrixP,
				      matrixk,
			       R_trainingimg,
			       T_trainingimg4x1);   
	T_trainingimg3x1=matrixk.inv()*MatrixP.col(3);
	T_trainingimg3x1.at<double>(0, 0) += 0;
	T_trainingimg3x1.at<double>(0, 1) -= 0;
	T_trainingimg3x1.at<double>(0, 2) -= 0;
	
	
	double rotationangle =10 * 3.1415 / 180;
	Mat rotation = RaboutY(rotationangle) * R_trainingimg;
	
	Mat projectionMatirx = Mat(3,4,CV_64FC1);
	for(int i =0; i < 3; i ++)
	{
		projectionMatirx.at<double>(0, i) = rotation.at<double>(0, i);
		projectionMatirx.at<double>(1, i) = rotation.at<double>(1, i);
		projectionMatirx.at<double>(2, i) = rotation.at<double>(2, i);
	}
	projectionMatirx.at<double>(0, 3) = T_trainingimg3x1.at<double>(0, 0);
	projectionMatirx.at<double>(1, 3) = T_trainingimg3x1.at<double>(1, 0);
	projectionMatirx.at<double>(2, 3) = T_trainingimg3x1.at<double>(2, 0);
	
	/*******************projection matrix estimation END*********************/
	MatrixP = matrixk*projectionMatirx;
	cout << MatrixP << endl;
	tm1.stop();
	double buildTime1 = tm1.getTimeMilli();
	cout << "RANSAC time: " << buildTime1 << " ms" << endl;  
	
	if(MatrixP.at<double>(2,3)==0)
		return MatrixP;
	else
	{    	
	    Mat tmpmat = imread(imageForShow);
	    
	    Mat mattmp = featurematching.project(matrices.matrix3d, MatrixP,  tmpmat);
	    // 
	    //    cout  << "camera pose is :\n"<<MatrixP << endl;
	    imwrite("result_"+ filenames.at(flag)+".jpg", mattmp);
	    
	    
	    vector<correspondingPair> corres;
	    string flename;
	    vector< pair<string, vector<correspondingPair> > > allcorres;
	    
	    for(int i =0; i < descriptors_trainingimages.size(); i++)
	    { 
		    vector<correspondingPair> correstmp;
		    TickMeter tm;
		    tm.start();
		    featurematching.MatchingStart2(128,
						    0.0,
						    descriptors_trainingimages.at(i),     // features of each training image
						    descriptors_testimages,   // features of test image
						    correstmp);  
		    descriptors_trainingimages.at(i).clear();
		    tm.stop();
		    double buildTime = tm.getTimeMilli();
		    cout << "total matching time: " << buildTime << " ms" << endl;
		    cout << filenames.at(flag) << endl; 
		    cout << trainingImagenames.at(i) << endl;
		    cout << correstmp.size() << endl;    
		    
		    
		    
		    //draw2Dkeypoints(correstmp, keypoints_testimages, imageForShow);
		    string trainimgtmp = trainingImagesKeysPath+trainingImagenames.at(i).substr(0, trainingImagenames.at(i).length()-4);
		    stringstream ss(trainimgtmp);
		    string sub_str;
		    while(getline(ss,sub_str,'/')) //以|为间隔分割test的内容
				continue;
		    string inputimagepath = trainimgtmp.substr(0, trainimgtmp.find_last_of(sub_str) - sub_str.length() + 1);
		    string filename = trainimgtmp.substr(trainimgtmp.find_last_of(sub_str)  - sub_str.length() + 1, trainimgtmp.length() -1);
		    
		    string zero = "0";
		    if(filename.length() < 8)
		    {		
			for(int a = 1; a < 8 - filename.length(); a ++)
			    zero += "0";
		    }
		    filename = zero + filename;						
		    string trainimg = inputimagepath + filename +".jpg";
		    string testimg = testImagesPath +testimagenames.at(flag);
		    cout << trainimg << "\n" << testimg << endl;
		    //drawmatching(trainimg, testimg, keypoints_testimages, correstmp);
		    pair<string, vector<correspondingPair> > pa;
		    pa.first = trainingImagenames.at(i);
		    pa.second = correstmp;
		    allcorres.push_back(pa);
	    }
// 	    for(int i = 0; i < allcorres.size(); i++)
// 	    {
// 		    cout  <<allcorres.at(i).second.size()<< endl;
// 	    }
	    
	    sort(allcorres.begin(), allcorres.end(), cmp);
	    Mat matforshow;
	    vector<int> distanceofpixel;
	    distanceofpixel.push_back(0);
	    distanceofpixel.push_back(5);
	    distanceofpixel.push_back(10);
	    distanceofpixel.push_back(20);
	    distanceofpixel.push_back(30);
	    distanceofpixel.push_back(50);
	    distanceofpixel.push_back(70);
	    distanceofpixel.push_back(90);
	    
	    for(int p = 0; p < topNearestImagesNum; p++)
	    {
		    cout  << allcorres.at(p).first << "\t" << allcorres.at(p).second.size() << " correspondences"<< endl;
		    for(int a = 0; a < distanceofpixel.size(); a++)
		    {
			    matforshow = featurematching.projection(matrices,
								    MatrixP, 
								    allcorres.at(p).second, 
								    keypoints_testimages, 
								    imageForShow,  
								    filenames.at(flag)+allcorres.at(p).first, 
								    distanceofpixel.at(a));	   
		    }
	    }
	    //Mat matforshow;
	    return matforshow;
	}
}


Mat changedetection::matchingStart2( vector<Point2d> keypoints_testimages,
				     vector<unsigned int> descriptors_testimages, 
				     vector<Point3d> keypoints_3Dpoints, 
				     vector<unsigned int> descriptors_3Dpoints,
				     mats  matrices,
				     string imageForShow,
				     int flag)
{
	
	//matching descriptors
	vector<correspondingPair> correspondings_origianl;  //两个descriptor文件的比较对应序列
	vector<correspondingPair> correspondings;
	
	
	TickMeter tm;
	tm.start();	    
	featurematching.MatchingStart(128,
				      0.0,
			       descriptors_testimages,     // features of each training image
			       descriptors_3Dpoints,   // features of test image
			       correspondings_origianl);  	    
	
	tm.stop();
	double buildTime = tm.getTimeMilli();
	cout << "Matching time: " << buildTime << " ms" << endl;
	
	vector<matchedPoints> matchedPointsfianl;
	
	//correspondings = featurematching.getCorrespondingsAfterErasingRepeats(correspondings_origianl);
	
	correspondings = correspondings_origianl;
	Mat matrix3Dpoints = Mat(4, correspondings.size(), CV_64F);
	for(int i =0; i < correspondings.size(); i ++)
	{
		
		matchedPoints onepair;
		onepair.x3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).x;
		onepair.y3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).y;
		onepair.z3d = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).z;
		onepair.x2d = keypoints_testimages.at(correspondings.at(i).indexInTraining).x;
		onepair.y2d = keypoints_testimages.at(correspondings.at(i).indexInTraining).y;
		matchedPointsfianl.push_back(onepair);
		matrix3Dpoints.at<double>(0, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).x;
		matrix3Dpoints.at<double>(1, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).y;
		matrix3Dpoints.at<double>(2, i) = keypoints_3Dpoints.at(correspondings.at(i).indexInTest).z;
		matrix3Dpoints.at<double>(3, i) = 1;
		
		//      cout << i<<"\t"<<onepair.x3d << "\t" 
		//     	       << onepair.y3d << "\t" 
		//     	       <<onepair.z3d<< "\t" 
		//     	       << onepair.x2d << "\t" 
		//     	       << onepair.y2d<< endl;
	}
	
	TickMeter tm1;
	tm1.start();	    
	/*******************projection matrix estimation BEGIN*********************/
	Mat MatrixP = pmat_ransac(matchedPointsfianl);
	cout << MatrixP << endl;
	/****to get intrinsic matrix****/
	Mat matrixk=Mat(3,3,CV_64FC1);	//intrinsic matrix
	Mat R_trainingimg=Mat(3,3,CV_64FC1);  
	Mat T_trainingimg4x1=Mat(4,1,CV_64FC1);
	Mat T_trainingimg3x1=Mat(3,1,CV_64FC1);
	cv::decomposeProjectionMatrix(MatrixP,
				      matrixk,
			       R_trainingimg,
			       T_trainingimg4x1);   
	T_trainingimg3x1=matrixk.inv()*MatrixP.col(3);
	T_trainingimg3x1.at<double>(0, 0) += 0;
	T_trainingimg3x1.at<double>(0, 1) -= 0;
	T_trainingimg3x1.at<double>(0, 2) -= 0;
	
	
	double rotationangle =5 * 3.1415 / 180;
	Mat rotation = RaboutY(rotationangle) * R_trainingimg;
	
	Mat projectionMatirx = Mat(3,4,CV_64FC1);
	for(int i =0; i < 3; i ++)
	{
		projectionMatirx.at<double>(0, i) = rotation.at<double>(0, i);
		projectionMatirx.at<double>(1, i) = rotation.at<double>(1, i);
		projectionMatirx.at<double>(2, i) = rotation.at<double>(2, i);
	}
	projectionMatirx.at<double>(0, 3) = T_trainingimg3x1.at<double>(0, 0);
	projectionMatirx.at<double>(1, 3) = T_trainingimg3x1.at<double>(1, 0);
	projectionMatirx.at<double>(2, 3) = T_trainingimg3x1.at<double>(2, 0);
	
	/*******************projection matrix estimation END*********************/
	MatrixP = matrixk*projectionMatirx;
	cout << MatrixP << endl;
	tm1.stop();
	double buildTime1 = tm1.getTimeMilli();
	cout << "RANSAC time: " << buildTime1 << " ms" << endl;  
	
	if(MatrixP.at<double>(2,3)==0)
		return MatrixP;
	else
	{    	
		
		Mat tmpmat = imread(imageForShow);
		
		Mat mattmp = featurematching.project(matrices.matrix3d, MatrixP,  tmpmat);
		// 
		//    cout  << "camera pose is :\n"<<MatrixP << endl;
		imwrite("result_"+ filenames.at(flag)+".jpg", mattmp);
		
		
		vector<correspondingPair> corres;
		string flename;
		vector< pair<string, vector<correspondingPair> > > allcorres;
		
		
		Mat mattrainimg =imread(trainingImagesKeysPath + trainingImagenames.at(0),CV_LOAD_IMAGE_GRAYSCALE);
		Mat mattestimg = imread(imageForShow,CV_LOAD_IMAGE_GRAYSCALE);
		// detecting keypoints
		SurfFeatureDetector detector(1000);
		vector<KeyPoint> keypoints1, keypoints2;
		detector.detect(mattrainimg, keypoints1);
		detector.detect(mattestimg, keypoints2);
		
		// computing descriptors
		SurfDescriptorExtractor extractor;
		Mat descriptors1, descriptors2;
		extractor.compute(mattrainimg, keypoints1, descriptors1);
		extractor.compute(mattestimg, keypoints2, descriptors2);
		
		// matching descriptors
		BruteForceMatcher<L2<float> > matcher;
		vector<DMatch> matches;
		matcher.match(descriptors1, descriptors2, matches);
		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < descriptors1.rows; i++ )
		{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
		}
		std::vector< DMatch > good_matches;

		for( int i = 0; i < descriptors1.rows; i++ )
		{ if( matches[i].distance < 7*min_dist )
		{ good_matches.push_back( matches[i]); }
		}

		
		namedWindow("matches", 1);
		Mat img_matches;
		drawMatches(mattrainimg, 
				keypoints1, 
				mattestimg, 
				keypoints2, 
				good_matches, 
				img_matches, 
				Scalar::all(-1), 0,
				vector<char>(), 0);	
		imshow("matches", img_matches);
		imwrite("matches.jpg",img_matches);
		//waitKey();
		vector<correspondingPair> correstmp;
		for(int si = 0; si < good_matches.size(); si ++)
		{
			correspondingPair copa;
			copa.indexInTest = good_matches.at(si).trainIdx;
			copa.indexInTraining = good_matches.at(si).queryIdx;
			correstmp.push_back(copa);
		}
		
		
// 		draw2Dkeypoints(correstmp, keypoints_testimages, imageForShow);
// 		string trainimg =trainingImagesKeysPath+trainingImagenames.at(i).substr(0, trainingImagenames.at(i).length()-4)+".jpg";
// 		string testimg = testImagesPath + filenames.at(flag).substr(0,filenames.at(flag).length()-4)+".jpg";
// 		drawmatching(trainimg, testimg, keypoints_testimages, correstmp);
		pair<string, vector<correspondingPair> > pa;
		pa.first = trainingImagenames.at(0);
		pa.second = correstmp;
		allcorres.push_back(pa);
		
		for(int i = 0; i < allcorres.size(); i++)
		{
			cout  <<allcorres.at(i).second.size()<< endl;
		}
		
		Mat matforshow;
		vector<int> distanceofpixel;
		//distanceofpixel.push_back(0);
		//distanceofpixel.push_back(5);
		//distanceofpixel.push_back(10);
		//distanceofpixel.push_back(20);
		//distanceofpixel.push_back(30);
		//distanceofpixel.push_back(50);
		distanceofpixel.push_back(70);
		//distanceofpixel.push_back(90);
		
			cout  << allcorres.at(0).first << "\t" << allcorres.at(0).second.size() <<"---------------------"<< endl;
			for(int a = 0; a < distanceofpixel.size(); a++)
			{
				matforshow = featurematching.projection2(matrices,
									MatrixP, 
									allcorres.at(0).second, 
									keypoints2, 
									imageForShow,  
									filenames.at(flag)+allcorres.at(0).first, 
									distanceofpixel.at(a));	   
			}
	
		//Mat matforshow;
		return matforshow;
	}
}

string changedetection::distanceofRotationAndTranslation(Mat orignalpose, vector<Mat> cameraposes, vector<string> trainingimages)
{
	vector<double> result1;
	vector<double> result2;
	
	Mat matrixkt=Mat(3,3,CV_64FC1);	//intrinsic matrix
	Mat R_t=Mat(3,3,CV_64FC1);  
	Mat T_t4x1=Mat(4,1,CV_64FC1);
	Mat T_t3x1=Mat(3,1,CV_64FC1);
	if(orignalpose.at<double>(2,3)==0)
	{
		exit(1);
	}
	cv::decomposeProjectionMatrix(orignalpose,
				      matrixkt,
			       R_t,
			       T_t4x1);
	//cout  << "true intrinsic: \n"<<matrixkt/matrixkt.at<double>(2,2) << endl;
	T_t3x1 = matrixkt.inv() * orignalpose.col(3);
	for(int i = 0; i < cameraposes.size(); i++)
	{
		//cout << cameraposes.at(i) << endl;
		//     if(cameraposes.at(i).at<double>(2,3)==0)
		//       continue;
		Mat matrix_hat = Mat(3,3,CV_64FC1);	//intrinsic matrix
		Mat R_hat = Mat(3,3,CV_64FC1);  
		Mat T_hat4x1 = Mat(4,1,CV_64FC1);
		Mat T_hat3x1 = Mat(3,1,CV_64FC1);
		//cout << poses26.at(i)<< endl;
		cv::decomposeProjectionMatrix(cameraposes.at(i),
					      matrix_hat,
				R_hat,
				T_hat4x1);
		//cout  << matrix_hat << endl;
		T_hat3x1 = matrix_hat.inv() * cameraposes.at(i).col(3);
		Mat R = R_t * R_hat.t();
		//cout << R << endl;
		double trace = R.at<double>(0, 0) +  R.at<double>(1, 1) + R.at<double>(2, 2);
		double theta = acos((trace - 1)/2);
		//cout << theta << endl;
		result1.push_back(theta);
		double errordis = norm(T_t3x1 - T_hat3x1)/norm(T_hat3x1);
		result2.push_back(errordis);    
		
	}
	//cout << "rotation change angles:" << endl;
	for(int i =0 ; i  <  result1.size(); i ++)
	{
		//cout  << trainingimages.at(i) <<": ";
	//	cout  << result1.at(i)  << endl;
	}
	//cout << "translation errors:" << endl;
	for(int i =0 ; i  <  result2.size(); i ++)
	{
		// cout  << trainingimages.at(i) <<": ";
	//	cout  << result2.at(i) << endl;    
	}  
	//cout << "n:" << endl;  
	ofstream outfile (string(testImagesPath+"positionerror").c_str());
	if(!outfile) 
	{
		cout << "!outfile"<< endl;
		exit(1);
	}     
	vector<float> resultn,resultntmp;
	for(int i =0 ; i  <  result1.size(); i ++)
	{
		//cout  << i <<": ";
		//cout  << result1.at(i)+ result2.at(i)  << endl;
		resultn.push_back(result1.at(i)+result2.at(i));
	}
	resultntmp=resultn;
	sort(resultn.begin(), resultn.end());
	for(int i =0 ; i  <  resultn.size(); i ++)
	{
		for(int j =0; j < resultntmp.size(); j ++)
		{
			if(resultn.at(i) == resultntmp.at(j))
			{
				outfile << j << ":\t" ;
				outfile << resultn.at(i) << endl;
				return trainingimages.at(j);
			}
		}
	}
	outfile.close(); 
}

string changedetection::experiment()
{
	string testimagepath = testImagesPath;
	string traininimagepath = trainingImagesKeysPath;
	string trainingimageposespath = "/media/36A831ACA8316C0D/Wave blocks Dataset/smallbloclsoutdoor/dataset3/training3Dpoints/ProjectionMatrices/";
	fileOperation fileoperation;
	
	
	vector<string> testimage = fileoperation.getFilenames(testimagepath, "*.pose");
	vector<Mat> testimageposes = fileoperation.getCameraPose(testimagepath,testimage,0);
	if(testimageposes.size()>1)
	{
		cout << "please set the number of test image as one.(in changedetection::experiment())"<< endl;
		exit(1);
	}
	vector<string> trainingimage = fileoperation.getFilenames(trainingimageposespath, "*.txt");
	vector<Mat> trainingimageposes = fileoperation.getCameraPose(trainingimageposespath,trainingimage,1);
	
	vector<string> trainingimages = fileoperation.getFilenames(traininimagepath, "*.jpg");
	
	cout  << trainingimageposes.size() << endl;
	string nearestimage;
	for(int i  =0; i < testimageposes.size(); i++)
	{
		cout  << testimageposes.at(i)<< endl;
		nearestimage = distanceofRotationAndTranslation(testimageposes.at(i), trainingimageposes,trainingimages);
		cout << "nearest image: " <<nearestimage << endl;
	}
	return nearestimage;
	//exit(1);
}

void changedetection::changeImageSize(const string trainingimagePath, const string resizedPath)
{
	fileOperation fileoperation;
	
	vector<string> imagename = fileoperation.getFilenames(trainingimagePath, "*.jpg");
	for(int i = 0; i < imagename.size(); i++)
	{
		Mat image = imread(trainingimagePath+ imagename.at(i));
		Mat imagetmp;
		resize(image, imagetmp, Size(image.cols/3, image.rows/3));
		imwrite(resizedPath + imagename.at(i),imagetmp);
	}
	cout << "size changing of images is done!" << endl;
	exit(1);
}


void changedetection::draw2Dkeypoints(vector<correspondingPair> correstmp,
				      vector<Point2d> keypoints_testimages,
				      string  imageForShow)
{
	static string count="1";
	Mat image =  imread(imageForShow);
	
	vector<Point2f> pts;
	int r2 = 5;
	for(int i = 0; i < correstmp.size(); i++)
	{
		Point2f pt;
		pt.x = (int)keypoints_testimages.at(correstmp.at(i).indexInTest).x;
		pt.y = (int)keypoints_testimages.at(correstmp.at(i).indexInTest).y;
		pts.push_back(pt);
		circle(image, pt, r2, CV_RGB(0, 255, 255), -1);          
	}
	for(int j = 0; j < keypoints_testimages.size(); j ++)
	{	
		Point2f p;
		p.x = (int)keypoints_testimages.at(j).x;
		p.y = (int)keypoints_testimages.at(j).y;
		int count=0;
		for(int i = 0; i < pts.size(); i++)
		{
			if(p.x != pts.at(i).x || p.y != pts.at(i).y)
				count ++;
		}
		if(count == pts.size())
			circle(image, p, r2, CV_RGB(255, 0, 0), -1);       
	}   
	imwrite(count+".jpg",image);
	count+="1";
}


Mat changedetection::RaboutX(const double& theta)
{
	Mat raboutx = (Mat_<double>(3,3) << 1,         0,          0,
		       0,cos(theta),-sin(theta),
		       0,sin(theta),cos(theta));
	cout<< raboutx<< endl;
	return raboutx;
}

Mat changedetection::RaboutY(const double& theta)
{
	Mat rabouty = (Mat_<double>(3,3) << cos(theta), 0, sin(theta),
		       0,          1,          0,
		-sin(theta),0, cos(theta));
	
	return rabouty;
}

Mat changedetection::RaboutZ(const double& theta)
{
	Mat raboutz = (Mat_<double>(3,3) << cos(theta),-sin(theta),0,
		       sin(theta), cos(theta),0,
		       0,       0,            1);
	
	return raboutz;
}















