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


#include "featurematching.h"



#define Npoints 8
#define th_reprojection_error 10.0000
#define th_inlire_rate 85.0
#define th_scal 1.0



FeatureMatching::FeatureMatching()
{
	countt=0;
}

FeatureMatching::~FeatureMatching()
{
	
}

string FeatureMatching::IntToString(int i)
//change int to string type
{
	stringstream strStream;
	strStream<<i;
	string s = strStream.str();
	return s;
}


bool FeatureMatching::readPt(ANNpoint p, 
			     const int dim, 
			     const vector<unsigned int> &descriptors,
			     const int nPts)
{    
	for (int i = 0; i < dim; i++) {      
		p[i] = descriptors.at(dim * nPts + i);
		//cout<<p[i]<<endl;
	}
	return true;
}

bool FeatureMatching::readPt_double(ANNpoint p, 
				    const int dim, 
				    const vector<double> &descriptors,
				    const int nPts)
{    
	for (int i = 0; i < dim; i++) {      
		p[i] = descriptors.at(dim * nPts + i);
		//cout<<p[i]<<endl;
	}
	return true;
}



int FeatureMatching::MatchingStart(//const int k,    // number of near neighbors
				   const int dim,  // dimension of feature vector
				   const double eps,  // imageForShow.rows
				   // const int maxPts,  // 
				   const vector<unsigned int> &descriptors,	// all feature vectors in a training image
				   // e.g., dim-dimensional vectores are concatenated into a single vector
				   const vector<unsigned int> &descriptorsQuery,// all feature vectors in a test image
				   vector<correspondingPair> &correspondes)
{
	
	int k = 2; // in this function, we use only 2 nearest neibors
	
	//
	// read i-th feature vectors from descriptors (training)
	// and store it to dataPts[i]
	//
	TickMeter tm;
	tm.start();	  
	int nPts = descriptors.size() / dim;		// number of feature vectors in a training image
	ANNpointArray dataPts = annAllocPts(nPts, dim);	// allocate data points
	for(int i = 0; i < nPts; i++)
		readPt(dataPts[i], dim, descriptors, i);		// read and store    
		
		//
		// build search structure
		//    
		// search structure
	ANNkd_tree	kdTree(dataPts,	// the data points
				nPts,		// number of points
				dim);		// dimension of space
	tm.stop();
	double buildTime = tm.getTimeMilli();
	cout << "KD tree construction time: " << buildTime << " ms" << endl;
	
	
	
	
	
	
	
	ANNpoint		queryPt = annAllocPt(dim);	// query point
	ANNidxArray		nnIdx = new ANNidx[k];		// near neighbor indices
	ANNdistArray	dists = new ANNdist[k];		// near neighbor distances
	
	
	int nPtsQuery = descriptorsQuery.size() / dim;	// number of feature vectors in the test image
	
	
	correspondes.resize(0);  // 
	for(int i = 0; i < nPtsQuery; i++)
	{   
		//
		// read i-th feature vectors from descriptors (test)
		// and store it to queryPt
		//
		readPt(queryPt, dim, descriptorsQuery, i); // read and store
		kdTree.annkSearch(						// search
		queryPt,					// query point
		2,						// number of near neighbors
		nnIdx,						// nearest neighbors (returned)
		dists,						// distances (returned)
		eps);						// error bound     
		
		// cout << dists[0] << "\t" << dists[1] << endl;
		
		
		
		if(dists[0] < 0.6 * dists[1])
		{ // tentativeMatch is OK
		// 	    if(dists[0] > 130000)
		// 		break; 
		correspondingPair tentativeMatch;
		tentativeMatch.indexInTest     = i;
		tentativeMatch.indexInTraining = nnIdx[0];
		
		correspondes.push_back(tentativeMatch);
		}
	}
	
	delete [] nnIdx;							// clean things up
	delete [] dists;
	annClose();								// done with ANN
	return 1;
}

int FeatureMatching::MatchingStart2(//const int k,    // number of near neighbors
				   const int dim,  // dimension of feature vector
				   const double eps,  // imageForShow.rows
				   // const int maxPts,  // 
				   const vector<unsigned int> &descriptors,	// all feature vectors in a training image
				   // e.g., dim-dimensional vectores are concatenated into a single vector
				   const vector<unsigned int> &descriptorsQuery,// all feature vectors in a test image
				   vector<correspondingPair> &correspondes)
{
	
	int k = 2; // in this function, we use only 2 nearest neibors
	
	//
	// read i-th feature vectors from descriptors (training)
	// and store it to dataPts[i]
	//
	TickMeter tm;
	tm.start();	  
	int nPts = descriptors.size() / dim;		// number of feature vectors in a training image
	ANNpointArray dataPts = annAllocPts(nPts, dim);	// allocate data points
	for(int i = 0; i < nPts; i++)
		readPt(dataPts[i], dim, descriptors, i);		// read and store    
		
		//
		// build search structure
		//    
		// search structure
	ANNkd_tree	kdTree(dataPts,	// the data points
				nPts,		// number of points
				dim);		// dimension of space
	tm.stop();
	double buildTime = tm.getTimeMilli();
	cout << "KD tree construction time: " << buildTime << " ms" << endl;
	
	
	
	
	
	
	
	ANNpoint		queryPt = annAllocPt(dim);	// query point
	ANNidxArray		nnIdx = new ANNidx[k];		// near neighbor indices
	ANNdistArray	dists = new ANNdist[k];		// near neighbor distances
	
	
	int nPtsQuery = descriptorsQuery.size() / dim;	// number of feature vectors in the test image
	
	
	correspondes.resize(0);  // 
	for(int i = 0; i < nPtsQuery; i++)
	{   
		//
		// read i-th feature vectors from descriptors (test)
		// and store it to queryPt
		//
		readPt(queryPt, dim, descriptorsQuery, i); // read and store
		kdTree.annkSearch(						// search
		queryPt,					// query point
		2,						// number of near neighbors
		nnIdx,						// nearest neighbors (returned)
		dists,						// distances (returned)
		eps);						// error bound     
		
		// cout << dists[0] << "\t" << dists[1] << endl;
		
		
		
		if(dists[0] < 0.6 * dists[1])
		{ // tentativeMatch is OK
		// 	    if(dists[0] > 130000)
		// 		break; 
		correspondingPair tentativeMatch;
		tentativeMatch.indexInTest     = i;
		tentativeMatch.indexInTraining = nnIdx[0];
		
		correspondes.push_back(tentativeMatch);
		}
	}
	
	delete [] nnIdx;							// clean things up
	delete [] dists;
	annClose();								// done with ANN
	return 1;
}

int FeatureMatching::MatchingStart_double(//const int k,    // number of near neighbors
					  const int dim,  // dimension of feature vector
					  const double eps,  // imageForShow.rows
					  // const int maxPts,  // 
					  const vector<double> &descriptors,	// all feature vectors in a training image
					  // e.g., dim-dimensional vectores are concatenated into a single vector
					  const vector<double> &descriptorsQuery,// all feature vectors in a test image
					  vector<correspondingPair> &correspondes)
{
	
	int k = 1; // in this function, we use only 2 nearest neibors
	
	//
	// read i-th feature vectors from descriptors (training)
	// and store it to dataPts[i]
	//
	
	int nPts = descriptors.size() / dim;		// number of feature vectors in a training image
	ANNpointArray dataPts = annAllocPts(nPts, dim);	// allocate data points
	for(int i = 0; i < nPts; i++)
		readPt_double(dataPts[i], dim, descriptors, i);		// read and store
		
		
		//
		// build search structure
		//
		
		// search structure
		ANNkd_tree*	kdTree = new ANNkd_tree(dataPts,	// the data points
							nPts,		// number of points
				      dim);		// dimension of space
		
		ANNpoint		queryPt = annAllocPt(dim);	// query point
		ANNidxArray		nnIdx = new ANNidx[k];		// near neighbor indices
		ANNdistArray	dists = new ANNdist[k];		// near neighbor distances
		
		
		int nPtsQuery = descriptorsQuery.size() / dim;	// number of feature vectors in the test image
		
		correspondes.resize(0);  // 
		
		for(int i = 0; i < nPtsQuery; i++)
		{
			
			
			
			//
			// read i-th feature vectors from descriptors (test)
			// and store it to queryPt
			//
			readPt_double(queryPt, dim, descriptorsQuery, i); // read and store
			
			kdTree->annkSearch(						// search
			queryPt,					// query point
			    1,						// number of near neighbors
			    nnIdx,						// nearest neighbors (returned)
			dists,						// distances (returned)
		eps);						// error bound
		
		
		
		//cout << dists[0] << endl; 
		
		correspondingPair tentativeMatch;
		tentativeMatch.indexInTest     = i;
		tentativeMatch.indexInTraining = nnIdx[0];
		
		correspondes.push_back(tentativeMatch);
		
		
		
		
		}
		
		delete [] nnIdx;							// clean things up
		delete [] dists;
		delete kdTree;
		annClose();								// done with ANN
		return 1;
}


static int vertex_cb(p_ply_argument argument)
{
	void *pdata;
	long indexCoord;
	
	ply_get_argument_user_data(argument, &pdata, &indexCoord);
	
	aPoint *points = *((aPoint**)pdata);
	
	long index;
	ply_get_argument_element(argument, NULL, &index);
	
	if (indexCoord == 1){
		points[index].x = ply_get_argument_value(argument);
	}
	if (indexCoord == 2){
		points[index].y = ply_get_argument_value(argument);
	}
	if (indexCoord == 3){
		points[index].z = ply_get_argument_value(argument);
	}
	
	return 1;
}

int FeatureMatching::readPLY(aPoint* points, aPoint* normal,  aPoint* color,const char* input_ply)
{
	p_ply ply = ply_open(input_ply, NULL);
	if (!ply) return 1;
	if (!ply_read_header(ply)) return 1;
	
	ply_set_read_cb(ply, "vertex", "x", vertex_cb, &points, 1);
	ply_set_read_cb(ply, "vertex", "y", vertex_cb, &points, 2);
	ply_set_read_cb(ply, "vertex", "z", vertex_cb, &points, 3);
	ply_set_read_cb(ply, "vertex", "nx", vertex_cb, &normal, 1);
	ply_set_read_cb(ply, "vertex", "ny", vertex_cb, &normal, 2);
	ply_set_read_cb(ply, "vertex", "nz", vertex_cb, &normal, 3);
	ply_set_read_cb(ply, "vertex", "diffuse_red", vertex_cb, &color, 1);
	ply_set_read_cb(ply, "vertex", "diffuse_green", vertex_cb, &color, 2);
	ply_set_read_cb(ply, "vertex", "diffuse_blue", vertex_cb, &color, 3);
	
	if (!points || !normal|| !color) {
		cout<<"Error: new aPoint.\n"<<endl;
		if(!points) delete [] points;
		if(!normal) delete [] normal;
		if(!color) delete [] color;
		return 1;
	}
	
	if (!ply_read(ply)) return 1;  // read entire data at once
	ply_close(ply);
	
	return 0;
}

int FeatureMatching::readPointnum(aPoint* points, aPoint* normal, const char* input_ply)
{
	int pointnum = 0;
	
	p_ply ply = ply_open(input_ply, NULL);
	if (!ply) return 1;
	if (!ply_read_header(ply)) return 1;
	
	long nvertices = 
	ply_set_read_cb(ply, "vertex", "x", vertex_cb, &points, 1);
	pointnum = int(nvertices);
	
	points = new aPoint[pointnum];	
	normal = new aPoint[pointnum];	
	
	ply_close(ply);
	
	return pointnum;
}


mats FeatureMatching::read3Dpoints(string plypath)
{
	
	aPoint* points = NULL;
	aPoint* normal = NULL;
	aPoint* color = NULL;
	int pointnum=0;
	pointnum = readPointnum(points, normal ,plypath.c_str());		//入力する点の総数を読み込む
	color  = new aPoint[pointnum];
	points = new aPoint[pointnum];			//3D座標	
	normal = new aPoint[pointnum];			//法線
	
	readPLY(points, normal, color, plypath.c_str());
	Mat matrix3D=Mat::zeros(4,pointnum,CV_64FC1);
	Mat matrixcolor = Mat::zeros(3, pointnum,CV_64FC1);
	Mat matrixnormal = Mat::zeros(3, pointnum,CV_64FC1);
	for(int i = 0; i < pointnum; i++){
		matrix3D.at<double>(0,i)=points[i].x;	
		matrix3D.at<double>(1,i)=points[i].y;	
		matrix3D.at<double>(2,i)=points[i].z;	
		matrix3D.at<double>(3,i)=1.0;	
		matrixcolor.at<double>(0,i) = color[i].x;
		matrixcolor.at<double>(1,i) = color[i].y;
		matrixcolor.at<double>(2,i) = color[i].z;
		matrixnormal.at<double>(0, i) = normal[i].x;
		matrixnormal.at<double>(1, i) = normal[i].y;
		matrixnormal.at<double>(2, i) = normal[i].z;
		//cout <<"nx:"<< normal[i].x << "ny:"<< normal[i].y << "nz:" << normal[i].z <<endl;
	}
	
	delete [] points;
	delete [] normal;
	delete [] color; 
	
	mats mat;
	mat.matrix3d= matrix3D;
	mat.matrixcolor=matrixcolor;
	mat.normal = matrixnormal;
	return mat;
}


Mat FeatureMatching::projection(Mat matrix3D,
				vector<Point2d> matchingpoints,
				Mat projectionMatrixP,
				Mat  imageForShow)
{
	Mat matrix2DreX = Mat(3, matrix3D.cols, CV_64FC1);
	matrix2DreX=projectionMatrixP*matrix3D;   
	
	Point2f pt;
	int r = 15;	//半径
	int pointSize = 30;
	int cou = 0;
	for(int i = 0; i < matrix2DreX.cols ; i++){
		matrix2DreX.at<double>(0,i)=matrix2DreX.at<double>(0,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(1,i)=matrix2DreX.at<double>(1,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(2,i)=matrix2DreX.at<double>(2,i)/matrix2DreX.at<double>(2,i);
		if(matrix2DreX.at<double>(0,i) > 0 && matrix2DreX.at<double>(1,i) > 0 && matrix2DreX.at<double>(0,i) <imageForShow.cols && matrix2DreX.at<double>(1,i) <imageForShow.rows)
		{
			for(int j =0; j < matchingpoints.size(); j++)
			{	
				double a = matchingpoints.at(j).x - matrix2DreX.at<double>(0,i);
				double b = matchingpoints.at(j).y -  matrix2DreX.at<double>(1,i);
				
				if(sqrt(a*a+b*b)<pointSize)
				{	   
					i ++;
					continue;
				}
			}
			
			pt.x = (int)matrix2DreX.at<double>(0,i);
			pt.y = (int)matrix2DreX.at<double>(1,i);
			cv::circle(imageForShow, pt, r, CV_RGB(0, 255, 0), -1);	
		}
	}  
	return imageForShow;
}



Mat FeatureMatching::project(Mat matrix3D,
			     Mat projectionMatrixP,
			     const Mat&  imageForShow)
{    
	Mat image = imageForShow;
	Mat matrix2DreX = Mat(3, matrix3D.cols, CV_64FC1);
	matrix2DreX=projectionMatrixP*matrix3D;
	Point2f pt;
	int r = 2;	//半径
	int cou = 0;
	
	for(int i = 0; i < matrix2DreX.cols ; i++){
		matrix2DreX.at<double>(0,i)=matrix2DreX.at<double>(0,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(1,i)=matrix2DreX.at<double>(1,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(2,i)=matrix2DreX.at<double>(2,i)/matrix2DreX.at<double>(2,i);
		if(matrix2DreX.at<double>(0,i) > 0 && matrix2DreX.at<double>(1,i) > 0 && matrix2DreX.at<double>(0,i) <image.cols && matrix2DreX.at<double>(1,i) <image.rows)
		{
			cou++;
			pt.x = (int)matrix2DreX.at<double>(0,i);
			pt.y = (int)matrix2DreX.at<double>(1,i);
			cv::circle(image, pt, r, CV_RGB(0, 255, 0), -1);
		}
	}  
	
	
	//     namedWindow("Sifttt",CV_WINDOW_AUTOSIZE);
	//     imshow("Sifttt", image);
	//     imwrite("result_"+flag+".jpg", image);
	return image;
}



Mat FeatureMatching::projection(Mat matrix3D,
				Mat pointsof3Dbeforethreshing,
				Mat projectionMatrixP,
				Mat  imageForShow)
{    
	Mat matrix2DreX = Mat(3, matrix3D.cols, CV_64FC1);
	matrix2DreX=projectionMatrixP*matrix3D;
	
	Mat matrix2DreXt = Mat(3, pointsof3Dbeforethreshing.cols, CV_64FC1);
	matrix2DreXt=projectionMatrixP*pointsof3Dbeforethreshing;
	
	Point2f pt;
	int r = 1;	//半径
	int pointSize = 15;
	
	int cou = 0;
	for(int i = 0; i < matrix2DreX.cols ; i++){
		matrix2DreX.at<double>(0,i)=matrix2DreX.at<double>(0,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(1,i)=matrix2DreX.at<double>(1,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(2,i)=matrix2DreX.at<double>(2,i)/matrix2DreX.at<double>(2,i);
		if(matrix2DreX.at<double>(0,i) > 0 && matrix2DreX.at<double>(1,i) > 0 && matrix2DreX.at<double>(0,i) <imageForShow.cols && matrix2DreX.at<double>(1,i) <imageForShow.rows)
		{
			int flag = 0;
			for(int j =0; j < matrix2DreXt.cols; j++)
			{
				matrix2DreXt.at<double>(0,j)=matrix2DreXt.at<double>(0,j)/matrix2DreXt.at<double>(2,j);
				matrix2DreXt.at<double>(1,j)=matrix2DreXt.at<double>(1,j)/matrix2DreXt.at<double>(2,j);
				matrix2DreXt.at<double>(2,j)=matrix2DreXt.at<double>(2,j)/matrix2DreXt.at<double>(2,j);
				double a = (matrix2DreXt.at<double>(0,j) - matrix2DreX.at<double>(0,i));
				double b = matrix2DreXt.at<double>(1,j) -  matrix2DreX.at<double>(1,i);
				
				// 	  if(sqrt(a*a+b*b)>10)
				// 	  {
					// 	    flag++;      
				// 	  }
				if(sqrt(a*a+b*b)<pointSize)
				{
					i ++;
					continue;
				}
			}
			pt.x = (int)matrix2DreX.at<double>(0,i);
			pt.y = (int)matrix2DreX.at<double>(1,i);
			cv::circle(imageForShow, pt, r, CV_RGB(0, 255, 0), -1);
			
		}
	}  
	
	
	//     namedWindow("Sifttt",CV_WINDOW_AUTOSIZE);
	//     imshow("Sifttt", imageForShow);
	//     imwrite("result_"+flag+".jpg", imageForShow);
	return imageForShow;
}

double FeatureMatching::getNormalVecAngle(Mat viewDirection, 
					  Mat normalVec)
{
	double angle;
	viewDirection = viewDirection / norm(viewDirection);
	normalVec = normalVec / norm(normalVec);
	angle = normalVec.dot(-viewDirection);
	return angle;
}


/**
 *  return projected 2D  points from 3D points after the normal vector calculation
 */
vector<correspondingFrom3dto2d> FeatureMatching::get2DpointsAfterNormalVec(mats matrix,
									   Mat projectiomMatrix,
									   Mat image,
									   int visibalAngle)
{
	
	/****to get intrinsic matrix****/
	Mat matrixk=Mat(3,3,CV_64FC1);	//intrinsic matrix
	Mat R_trainingimg=Mat(3,3,CV_64FC1);  
	Mat T_trainingimg4x1=Mat(4,1,CV_64FC1);
	Mat T_trainingimg3x1=Mat(3,1,CV_64FC1);
	cv::decomposeProjectionMatrix(projectiomMatrix,
				      matrixk,
			       R_trainingimg,
			       T_trainingimg4x1);   
	
	
	/***projected 2D points***/
	vector<Point3d> vec2D;
	vector<Point3d> vec3D;
	vector<correspondingFrom3dto2d> points;
	
	Mat imageForShow = image;
	for(unsigned int i=0; i<matrix.matrix3d.cols; i++)
	{
		
		Mat onecol = Mat(3, 1, CV_64FC1);
		onecol.at<double>(0, 0) = matrix.normal.at<double>(0, i);
		onecol.at<double>(1, 0) = matrix.normal.at<double>(1, i);
		onecol.at<double>(2, 0) = matrix.normal.at<double>(2, i);
		Mat viewDirection = R_trainingimg.row(2).t();    
		/***return the cos() value of angle which between view direction and normal vector***/
		double returnofDot = getNormalVecAngle(viewDirection,
						       onecol);
		//     cout <<  acos(returnofDot)*180/3.1415926 << endl;
		if(returnofDot > cos(visibalAngle*3.1415/180))
		{
			
			correspondingFrom3dto2d point;
			point.point2d.x = Mat(projectiomMatrix * matrix.matrix3d.col(i)).at<double>(0, 0)/Mat(projectiomMatrix * matrix.matrix3d.col(i)).at<double>(2, 0);
			point.point2d.y = Mat(projectiomMatrix * matrix.matrix3d.col(i)).at<double>(1, 0)/Mat(projectiomMatrix * matrix.matrix3d.col(i)).at<double>(2, 0);
			point.point2d.z = Mat(projectiomMatrix * matrix.matrix3d.col(i)).at<double>(2, 0)/Mat(projectiomMatrix * matrix.matrix3d.col(i)).at<double>(2, 0);
			
			point.point3d.x = matrix.matrix3d.at<double>(0, i);
			point.point3d.y = matrix.matrix3d.at<double>(1, i);
			point.point3d.z = matrix.matrix3d.at<double>(2, i);
			
			point.normalVec = Mat(3,1,CV_64FC1);
			point.normalVec.at<double>(0, 0) = matrix.normal.at<double>(0, i);
			point.normalVec.at<double>(1, 0) = matrix.normal.at<double>(1, i);
			point.normalVec.at<double>(2, 0) = matrix.normal.at<double>(2, i);
			
			if(point.point2d.x>0 
				&& point.point2d.y>0
				&& point.point2d.y < imageForShow.cols 
				&& point.point2d.y < imageForShow.rows)
			{
				points.push_back(point);
			}      
		}
	}
	
	return points;
}

Mat FeatureMatching::projection(mats matrix3D,
				Mat projectionMatrixP,
				vector<correspondingPair> corres,
				vector<Point2d> keypoints_testimages,
				string  imageForShow,
				string filename,
				int distanceofpixel )
{
	
	
	
	
	Mat image =  imread(imageForShow);
	Mat imagecorres = imread(imageForShow);
	Mat imagetmpresult = imread(imageForShow);
// 	Mat imagemask = imread("maskimg.jpg",1);
// 	if(imagemask.empty())
// 	{
// 		cout << "mask image has not been set up yet!" <<  endl;
// 		imagemask = Mat::zeros(image.rows, image.cols, image.type());
// 	}
	vector<correspondingFrom3dto2d> points  = get2DpointsAfterNormalVec(matrix3D, projectionMatrixP, image, 60); 
	//cout << points.size() << endl;
	Mat matrix2DreX = Mat(3, points.size(), CV_64FC1);
	//matrix2DreX=projectionMatrixP*matrix3D.matrix3d;
	for(int i = 0; i < points.size(); i++)
	{
		matrix2DreX.at<double>(0, i) = points.at(i).point2d.x;
		matrix2DreX.at<double>(1, i) = points.at(i).point2d.y;
		matrix2DreX.at<double>(2, i) = points.at(i).point2d.z;
	}
	Point2f pt;
	Point2f p;
	int r = 2;	//半径
	int r2 = 5;
	int pointSize = distanceofpixel;
	int count_inlier = 0;
	int count_outlier = 0;
	
	vector<correspondingPair> correstmp;
	vector<correspondingPair>::iterator iter;
	for(iter = corres.begin(); iter != corres.end(); iter++)
	{
		pt.x = (int)keypoints_testimages.at((*iter).indexInTest).x;
		pt.y = (int)keypoints_testimages.at((*iter).indexInTest).y;
		//       if (imagemask.at<Vec3b>(pt.y, pt.x)[0] < 255 && imagemask.at<Vec3b>(pt.y, pt.x)[1] < 255 &&imagemask.at<Vec3b>(pt.y, pt.x)[2] < 255)
		//       {
			// 	//corres.erase(iter);
		// 	//cout << "erase" << keypoints_testimages.at((*iter).indexInTest).x << " " << keypoints_testimages.at((*iter).indexInTest).y <<endl;
		// 	continue;
		//       }
		correstmp.push_back(*iter);
	}
	//cout  << corres.size()<<" " << correstmp.size() << endl;
	for(int i = 0; i < correstmp.size(); i++)
	{
		pt.x = (int)keypoints_testimages.at(correstmp.at(i).indexInTest).x;
		pt.y = (int)keypoints_testimages.at(correstmp.at(i).indexInTest).y;
		circle(imagecorres, pt, r2, CV_RGB(0, 255, 255), -1);
		circle(imagetmpresult, pt, r2, CV_RGB(0, 255, 255), -1);       
	}   
	
	
	for(int i = 0; i < matrix2DreX.cols ; i++)
	{
		matrix2DreX.at<double>(0,i)=matrix2DreX.at<double>(0,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(1,i)=matrix2DreX.at<double>(1,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(2,i)=matrix2DreX.at<double>(2,i)/matrix2DreX.at<double>(2,i);
		if(matrix2DreX.at<double>(0,i) > 0 && matrix2DreX.at<double>(1,i) > 0 && matrix2DreX.at<double>(0,i) <image.cols && matrix2DreX.at<double>(1,i) <image.rows)
		{
			int flag = 0;
			for(int j =0; j < correstmp.size(); j++)
			{
				double a = keypoints_testimages.at(correstmp.at(j).indexInTest).x - matrix2DreX.at<double>(0,i);
				double b = keypoints_testimages.at(correstmp.at(j).indexInTest).y -  matrix2DreX.at<double>(1,i);
				
				if(sqrt(a*a+b*b)<pointSize)
				{
					// i++;
					flag++;
					// continue;
				}
			}
			if(flag ==0)
			{
				pt.x = (int)matrix2DreX.at<double>(0,i);
				pt.y = (int)matrix2DreX.at<double>(1,i);
				cv::circle(image, pt, r, CV_RGB(255, 0, 0), -1);
				cv::circle(imagetmpresult, pt, r, CV_RGB(255, 0, 0), -1);
// 				if (imagemask.at<Vec3b>(pt.y, pt.x)[0] < 255 && imagemask.at<Vec3b>(pt.y, pt.x)[1] < 255 &&imagemask.at<Vec3b>(pt.y, pt.x)[2] < 255)
// 				{
// 					// cv::circle(image, pt, r, CV_RGB(255, 0, 0), -1);
// 					count_inlier++;	    
// 				}
// 				else
// 				{
// 					count_outlier++;
// 				}
			}
		}
	}  
// 	cout   << count_inlier << endl;
// 	cout   << count_outlier << endl;
	namedWindow("Sifttt",CV_WINDOW_AUTOSIZE);   
	imwrite("result_"+filename+IntToString(distanceofpixel)+".jpg", image);
	imwrite("result_"+filename+IntToString(distanceofpixel)+"_tmp.jpg", imagetmpresult);
	imwrite("result_"+filename+"corres.jpg", imagecorres);  
	return image;
}

Mat FeatureMatching::projection2(mats matrix3D,
				Mat projectionMatrixP,
				vector<correspondingPair> corres,
				vector<KeyPoint> keypoints_testimages,
				string  imageForShow,
				string filename,
				int distanceofpixel )
{
	
	
	
	
	Mat image =  imread(imageForShow);
	Mat imagecorres = imread(imageForShow);
	Mat imagetmpresult = imread(imageForShow);
// 	Mat imagemask = imread("maskimg.jpg",1);
// 	if(imagemask.empty())
// 	{
// 		cout << "mask image has not been set up yet!" <<  endl;
// 		exit(1);
// 	}
	vector<correspondingFrom3dto2d> points  = get2DpointsAfterNormalVec(matrix3D, projectionMatrixP, image, 60); 
	//cout << points.size() << endl;
	Mat matrix2DreX = Mat(3, points.size(), CV_64FC1);
	//matrix2DreX=projectionMatrixP*matrix3D.matrix3d;
	for(int i = 0; i < points.size(); i++)
	{
		matrix2DreX.at<double>(0, i) = points.at(i).point2d.x;
		matrix2DreX.at<double>(1, i) = points.at(i).point2d.y;
		matrix2DreX.at<double>(2, i) = points.at(i).point2d.z;
	}
	Point2f pt;
	Point2f p;
	int r = 2;	//半径
	int r2 = 5;
	int pointSize = distanceofpixel;
	int count_inlier = 0;
	int count_outlier = 0;
	
	vector<correspondingPair> correstmp;
	vector<correspondingPair>::iterator iter;
	for(iter = corres.begin(); iter != corres.end(); iter++)
	{
		pt.x = (int)keypoints_testimages.at((*iter).indexInTest).pt.x;
		pt.y = (int)keypoints_testimages.at((*iter).indexInTest).pt.y;
		//       if (imagemask.at<Vec3b>(pt.y, pt.x)[0] < 255 && imagemask.at<Vec3b>(pt.y, pt.x)[1] < 255 &&imagemask.at<Vec3b>(pt.y, pt.x)[2] < 255)
		//       {
			// 	//corres.erase(iter);
		// 	//cout << "erase" << keypoints_testimages.at((*iter).indexInTest).x << " " << keypoints_testimages.at((*iter).indexInTest).y <<endl;
		// 	continue;
		//       }
		correstmp.push_back(*iter);
	}
	//cout  << corres.size()<<" " << correstmp.size() << endl;
	for(int i = 0; i < correstmp.size(); i++)
	{
		pt.x = (int)keypoints_testimages.at(correstmp.at(i).indexInTest).pt.x;
		pt.y = (int)keypoints_testimages.at(correstmp.at(i).indexInTest).pt.y;
		circle(imagecorres, pt, r2, CV_RGB(0, 255, 255), -1);
		circle(imagetmpresult, pt, r2, CV_RGB(0, 255, 255), -1);       
	}   
	
	
	for(int i = 0; i < matrix2DreX.cols ; i++)
	{
		matrix2DreX.at<double>(0,i)=matrix2DreX.at<double>(0,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(1,i)=matrix2DreX.at<double>(1,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(2,i)=matrix2DreX.at<double>(2,i)/matrix2DreX.at<double>(2,i);
		if(matrix2DreX.at<double>(0,i) > 0 && matrix2DreX.at<double>(1,i) > 0 && matrix2DreX.at<double>(0,i) <image.cols && matrix2DreX.at<double>(1,i) <image.rows)
		{
			int flag = 0;
			for(int j =0; j < correstmp.size(); j++)
			{
				double a = keypoints_testimages.at(correstmp.at(j).indexInTest).pt.x - matrix2DreX.at<double>(0,i);
				double b = keypoints_testimages.at(correstmp.at(j).indexInTest).pt.y -  matrix2DreX.at<double>(1,i);
				
				if(sqrt(a*a+b*b)<pointSize)
				{
					// i++;
					flag++;
					// continue;
				}
			}
			if(flag ==0)
			{
				pt.x = (int)matrix2DreX.at<double>(0,i);
				pt.y = (int)matrix2DreX.at<double>(1,i);
				cv::circle(image, pt, r, CV_RGB(255, 0, 0), -1);
				cv::circle(imagetmpresult, pt, r, CV_RGB(255, 0, 0), -1);
// 				if (imagemask.at<Vec3b>(pt.y, pt.x)[0] < 255 && imagemask.at<Vec3b>(pt.y, pt.x)[1] < 255 &&imagemask.at<Vec3b>(pt.y, pt.x)[2] < 255)
// 				{
// 					// cv::circle(image, pt, r, CV_RGB(255, 0, 0), -1);
// 					count_inlier++;	    
// 				}
// 				else
// 				{
// 					count_outlier++;
// 				}
			}
		}
	}  
// 	cout   << count_inlier << endl;
// 	cout   << count_outlier << endl;
	namedWindow("Sifttt",CV_WINDOW_AUTOSIZE);   
	imwrite("result_"+filename+IntToString(distanceofpixel)+".jpg", image);
	imshow("detection result", image);
//	waitKey();
	imwrite("result_"+filename+IntToString(distanceofpixel)+"_tmp.jpg", imagetmpresult);
	imwrite("result_"+filename+"corres.jpg", imagecorres);  
	return image;
}


/**
 * project 3D points onto 2D images without norm vector as 60.  only for viewing. 
 */
int FeatureMatching::projection (Mat matrix3D,
				 Mat matrixcolor,
				 Mat projectionMatrix,
				 string testimage)
{
	//cout  << "\nmatrix3D.rows:\n" << matrix3D.rows <<"\t"<< "\nmatrix3D.cols :\n" <<matrix3D.cols <<endl;
	
	Mat matrix2DreX=projectionMatrix * matrix3D;
	Point2f pt;
	int cou = 0;    
	Mat imageForShow = imread(testimage);
	Mat_<Vec3b> fisheye_im(imageForShow.rows, imageForShow.cols, Vec3b(0,0,0));
	
	mats matrices;
	matrices.matrix3d = matrix3D;
	matrices.matrixcolor = matrixcolor;
	
	
	//cout<<fisheye_im.cols<<"\t"<<fisheye_im.rows<<endl;
	for(int i = 0; i < matrix2DreX.cols ; i++){
		matrix2DreX.at<double>(0,i)=matrix2DreX.at<double>(0,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(1,i)=matrix2DreX.at<double>(1,i)/matrix2DreX.at<double>(2,i);
		matrix2DreX.at<double>(2,i)=matrix2DreX.at<double>(2,i)/matrix2DreX.at<double>(2,i);
		
		if(matrix2DreX.at<double>(0,i) > 0 
			&& matrix2DreX.at<double>(1,i) > 0 
			&& matrix2DreX.at<double>(0,i) <imageForShow.cols 
			&& matrix2DreX.at<double>(1,i) <imageForShow.rows)
		{
			cou++;
			pt.x = (int)matrix2DreX.at<double>(0,i);
			pt.y = (int)matrix2DreX.at<double>(1,i);
			fisheye_im(pt) = Vec3b(matrixcolor.at<double>(0,i),
					       matrixcolor.at<double>(1,i),
					       matrixcolor.at<double>(2,i));
			//circle(imageForShow, pt, 1, CV_RGB(0, 255, 0), -1);
		}
	}
	namedWindow("Sifttt",CV_WINDOW_AUTOSIZE);
	imshow("Sifttt", fisheye_im);
	imwrite("result_color.jpg", fisheye_im);
	
}


vector<correspondingPair> FeatureMatching::getCorrespondingsAfterErasingRepeats(vector<correspondingPair> correspondings)
{
	vector<correspondingPair> correspondings_erase_repeats;
	int flag = 0;
	for(unsigned int i = 0; i < correspondings.size(); i++){
		flag = 0;
		for(unsigned int j = 0; j < correspondings.size(); j++){
			if(j != i){
				if(correspondings.at(i).indexInTest == correspondings.at(j).indexInTest || correspondings.at(i).indexInTraining == correspondings.at(j).indexInTraining){
					cerr << ".";
					flag = 1;
				}
			}
		}
		if(flag == 0){
			correspondings_erase_repeats.push_back(correspondings.at(i));
		}
	}
	
	return correspondings_erase_repeats;
}

