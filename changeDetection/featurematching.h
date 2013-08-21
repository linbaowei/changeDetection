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


#ifndef FEATUREMATCHING_H
#define FEATUREMATCHING_H




#include <ctime>
#include <set>
#include <fstream>
#include <iostream>
using namespace std;


#include "rply.h"
#include <ANN/ANN.h>					// ANN declarations
#include <vector>
#include "GL/glew.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/GlobalUtil.h"
#include "cv.h"
#include "highgui.h"
#include "opencv2/contrib/contrib.hpp"
using namespace cv;


typedef struct {
	double x;
	double y;
	double z;
}	aPoint;

class matchedPoints{
public:
      double x3d;
      double y3d;
      double z3d;
      double x2d;
      double y2d;
};


typedef struct {
    int indexInTraining;
    int indexInTest;
} correspondingPair;

struct mats{
	Mat matrix3d;
	Mat normal;
	Mat matrixcolor;
};


struct correspondingFrom3dto2d
{
  Point3d point3d;
  Point3d point2d;
  Mat normalVec;
};

class FeatureMatching
{
      
public:
    int		_k;			// number of nearest neighbors
    int		_dim;			// dimension
    double	_eps;			// error bound
    int		_maxPts;		// maximum number of data points
    int 	countt;
    
    // read point (false on EOF)
    bool readPt(ANNpoint p, 
		const int dim, 
		const vector<unsigned int> &descriptors,
		const int nPts);
    
     
    
    void printPt(ANNpoint p, int dim);								// print point
    
    //begin to match
    int MatchingStart(//const int k, 
		      const int dim, 
		      const double eps, 
		      //const int maxPts,  
		      const vector<unsigned int> &descriptors,
		      const vector<unsigned int> &descriptorsQuery,
		      vector<correspondingPair> &correspondes);
    
     //begin to match
    int MatchingStart2(//const int k, 
		      const int dim, 
		      const double eps, 
		      //const int maxPts,  
		      const vector<unsigned int> &descriptors,
		      const vector<unsigned int> &descriptorsQuery,
		      vector<correspondingPair> &correspondes);
    
    bool readPt_double(ANNpoint p, 
		const int dim, 
		const vector<double> &descriptors,
		const int nPts);
    
     //begin to match
    int MatchingStart_double(//const int k,    // number of near neighbors
				   const int dim,  // dimension of feature vector
				   const double eps,  // imageForShow.rows
				   // const int maxPts,  // 
				   const vector<double> &descriptors,	// all feature vectors in a training image
				   // e.g., dim-dimensional vectores are concatenated into a single vector
				   const vector<double> &descriptorsQuery,// all feature vectors in a test image
				   vector<correspondingPair> &correspondes);
    
    Mat projection(Mat matrix3D,
				 vector<Point2d> matchingpoints,
				Mat projectionMatrixP,
				Mat  imageForShow);
    
    Mat project(Mat matrix3D,
				Mat projectionMatrixP,
				const Mat&  imageForShow);
    
    Mat projection(Mat matrix3D,
			    Mat pointsof3Dbeforethreshing,
				Mat projectionMatrixP,
				Mat  imageForShow);    
    int readPointnum(aPoint *points,
		aPoint *normal, 
		const char *input_ply);//read the total number of ply file.
    int readPLY(aPoint *points, 
		aPoint *normal,
		aPoint *color,
		const char *input_ply);
    mats read3Dpoints(string plypath);
    string IntToString(int i); 
    int projection (Mat matrix3D,
	Mat matrixcolor,
	Mat projectionMatrix,
	string testimage);
    Mat projection(mats matrix3D,
				Mat projectionMatrixP,
				vector<correspondingPair> corres,
				vector<Point2d> keypoints_testimages,
				string  imageForShow,
				string filename,
				int p);
    
    Mat projection2(mats matrix3D,
				Mat projectionMatrixP,
				vector<correspondingPair> corres,
				vector<KeyPoint> keypoints_testimages,
				string  imageForShow,
				string filename,
				int p);
    vector<correspondingPair> getCorrespondingsAfterErasingRepeats(vector<correspondingPair> correspondings);
    double getNormalVecAngle(Mat viewDirection, 
					   Mat normalVec);
    
    vector<correspondingFrom3dto2d> get2DpointsAfterNormalVec(mats matrix,
									    Mat projectiomMatrix,
									    Mat image, 
									    int visibalAngle);
    
    
    FeatureMatching();
    virtual ~FeatureMatching();
};

#endif // FEATUREMATCHING_H
