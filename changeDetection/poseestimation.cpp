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




#include "poseestimation.h"
#include "fileoperation.h"

poseEstimation::poseEstimation ()
{

}

poseEstimation::~poseEstimation ()
{

}

int poseEstimation::cameraCalibration (Mat& intrinsic, Mat& distCoeffs)
{
   int numBoards = 30;
    int numCornersHor = 9;
    int numCornersVer = 6;

    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);

    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points;

    vector<Point2f> corners;
    int successes = 0;

    Mat image;
    Mat gray_image;

    vector<Point3f> obj;
    for (int j = 0; j < numSquares; j++) {
        obj.push_back(Point3f(j / numCornersHor, j % numCornersHor, 0.0f));
    }
    
    fileOperation fileoperation;
    vector<string> imagenames = fileoperation.getFilenames("./calibration/","*.jpg");
    vector<Mat> images;
    for(int i = 0; i < imagenames.size(); i ++)
    {      
      Mat image = imread(string("./calibration/"+imagenames.at(i)).c_str());
      images.push_back(image);
    }
    for (int i = 1; i < images.size(); i++) 
    {
        image = images.at(i);
        cvtColor(image, gray_image, CV_BGR2GRAY);

        bool found = findChessboardCorners(image, board_sz, corners,
                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found) {
            cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1),
                    TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(gray_image, board_sz, Mat(corners), found);
        }
// 	Mat outimage;
// 	resize(gray_image,outimage,Size(1000,800));
//         //imshow("win1", outimage);
//         imshow("win2",outimage);
//         int key = waitKey(0);
//         if (key == 27)
//             return 0;

        if (found != 0) {
            cout << "test" << endl;
            image_points.push_back(corners);
            object_points.push_back(obj);
            printf("Snap stored!\n");

            successes++;

            if (successes >= numBoards)
                break;
                    // When X is greater than 4, calibrateCamera call segfaults
        }
        printf("at loop %d, result: %d   capacity:[%d %d]\n", i, found,
                image_points.capacity(), object_points.capacity());

    }

    intrinsic = Mat(3, 3, CV_64FC1);
    vector<Mat> rvecs;
    vector<Mat> tvecs;

    intrinsic.ptr<float> (0)[0] = 1;
    intrinsic.ptr<float> (1)[1] = 1;

    cout << "calling calibrate BEGIN---------\n\n" << endl;
    calibrateCamera(object_points, image_points, image.size(), intrinsic,
            distCoeffs, rvecs, tvecs);
    cout << "Finished" << endl;

    return 0;
  
}
   
   
Mat poseEstimation::findPose(vector<matchedPoints> matchedPointsfianl, Mat intricsic, Mat distCoeffs, Mat& rvec, Mat& tvec)
{
  vector<Point3f> objectpoints;
  vector<Point2f> imagepoints;
  for(int i  = 0; i < matchedPointsfianl.size(); i ++)
  {
    Point3f point;
    point.x = matchedPointsfianl.at(i).x3d;
    point.y = matchedPointsfianl.at(i).y3d;
    point.z = matchedPointsfianl.at(i).z3d;
    objectpoints.push_back(point);
    Point2f point2;
    point2.x = matchedPointsfianl.at(i).x2d;
    point2.y = matchedPointsfianl.at(i).y2d;
    imagepoints.push_back(point2);
  }
  solvePnPRansac(objectpoints, imagepoints, intricsic, distCoeffs, rvec, tvec);
  
  Mat rotation;
  Rodrigues(rvec, rotation);
  cout << "rotaion: \n" << rotation << endl;
  cout  << "translation: \n " << tvec << endl;
  Mat projectionMatirx = Mat(3,4,CV_64FC1);
  for(int i =0; i < 3; i ++)
  {
    projectionMatirx.at<double>(0, i) = rotation.at<double>(0, i);
    projectionMatirx.at<double>(1, i) = rotation.at<double>(1, i);
    projectionMatirx.at<double>(2, i) = rotation.at<double>(2, i);
  }
    projectionMatirx.at<double>(0, 3) = tvec.at<double>(0, 0);
    projectionMatirx.at<double>(1, 3) = tvec.at<double>(1, 0);
    projectionMatirx.at<double>(2, 3) = tvec.at<double>(2, 0);
    
    Mat camerapose = intricsic*projectionMatirx;
    cout << "projection Matirx:\n" << camerapose/camerapose.at<double>(2,3) << endl;
  return camerapose;
}
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   