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



#include "options.h"
#include "readFeatures.h"


void readAllFeatureFiles(const string inputFilename,
			 vector<string> &allFilenames){

  // read all files in the input file

  ifstream ifs (inputFilename.c_str() );
  if (!ifs) {
    cerr << "Cannot open " << inputFilename << endl;
    exit(1);
  }
  string siftFile;
  while (ifs.good()) {
    ifs >> siftFile;
    if (!ifs.good()) break;
    allFilenames.push_back( siftFile );
  }
  ifs.close();
}


void readFeaturestoVec(const string inputFilename,
				    vector<Point2d> &keypointsCoor, 
		                    vector<unsigned int> &descriptor)
{
   ifstream ifs ( inputFilename.c_str() );
  if (!ifs) {
    cerr << "Cannot open " << inputFilename << endl;
    exit(1);
  }

  for (int j = 0; j <2; j++) {
    double num;
    ifs >> num; // just skip
    //cout << num <<" are skipped"<< endl;
  }

  vector < vector<int> > trainingDataVector;
  Point2d point;
  while (ifs.good()) {

    for (int j = 0; j < 4; j++) {
      double num;
      ifs >> num; 
      if(j == 0)
      {
	point.y = num;
      }
      if(j == 1)
      {
	point.x = num;	
      }	
    }
    if (!ifs.good()) break;

    vector<int> feature;
    for(unsigned int j = 0; j < 128; j++) {
      int k;
      ifs >> k;
      if (!ifs.good())
      {
       if (j == 0) break;
	cerr << "Error while reading " << inputFilename << endl;
	exit(1);
      }
      feature.push_back(k);
    } // for j

    if (!ifs.good()) break;
    keypointsCoor.push_back(point);
    trainingDataVector.push_back(feature);

  } // end of while


  //
  // copy all features to the argument
  //
  //descriptor = Mat(trainingDataVector.size(), 128, CV_64FC1);
  try {
   for(int i =0; i<trainingDataVector.size(); i++)
   {
     for(int j = 0; j < 128; j ++)
     {
       descriptor.push_back( trainingDataVector.at(i).at(j));
    }
  }
  } catch (bad_alloc e) {
    cerr << "memory allocate error" << endl;
    exit(1);
  }
  
  ifs.close();
}


void read3DFeaturestoVec(const string inputFilename,
				    vector<Point3d> &keypointsCoor,
				    vector<unsigned int> &descriptor)
{
    ifstream ifs ( inputFilename.c_str() );
  if (!ifs) {
    cerr << "Cannot open " << inputFilename << endl;
    exit(1);
  }

  for (int j = 0; j <0; j++) {
    double num;
    ifs >> num; // just skip
  }

  vector < vector<int> > trainingDataVector;
  Point3d point;
  while (ifs.good()) {

    for (int j = 0; j < 8; j++) {
      double num;
      ifs >> num; // just skip
      if(j == 0)
	point.x = num;
      if(j == 1)
	point.y = num;
      if(j == 2)
      {
	point.z = num;	
      }
    }
    if (!ifs.good()) break;

    vector<int> feature;
    for(unsigned int j = 0; j < 128; j++) {
      int k;
      ifs >> k;
      if (!ifs.good()) {
       if (j == 0) break;
	cerr << "Error while reading " << inputFilename << endl;
	exit(1);
      }
      feature.push_back(k);
    } // for j

    if (!ifs.good()) break;
    keypointsCoor.push_back(point);
    trainingDataVector.push_back(feature);

  } // end of while


  //
  // copy all features to the argument
  //
  //descriptor = Mat(trainingDataVector.size(), 128, CV_64FC1);
  try {
   for(int i =0; i<trainingDataVector.size(); i++)
   {
     for(int j = 0; j < 128; j ++)
     {
       descriptor.push_back( trainingDataVector.at(i).at(j));
    }
  }
  } catch (bad_alloc e) {
    cerr << "memory allocate error" << endl;
    exit(1);
  }
  
  ifs.close();
}


void readFeaturestoMat(const string inputFilename,
				    Mat &descriptor)
{
   ifstream ifs ( inputFilename.c_str() );
  if (!ifs) {
    cerr << "Cannot open " << inputFilename << endl;
    exit(1);
  }

  for (int j = 0; j <2; j++) {
    double num;
    ifs >> num; // just skip
  }

  vector < vector<int> > trainingDataVector;

  while (ifs.good()) {

    for (int j = 0; j < 4; j++) {
      double num;
      ifs >> num; // just skip
    }
    if (!ifs.good()) break;

    vector<int> feature;
    for(unsigned int j = 0; j < 128; j++) {
      int k;
      ifs >> k;
      if (!ifs.good()) {
       if (j == 0) break;
	cerr << "Error while reading " << inputFilename << endl;
	exit(1);
      }
      feature.push_back(k);
    } // for j

    if (!ifs.good()) break;
    trainingDataVector.push_back(feature);

  } // end of while


  //
  // copy all features to the argument
  //
  descriptor = Mat(trainingDataVector.size(), 128, CV_64FC1);
  try {
   for(int i =0; i<trainingDataVector.size(); i++)
   {
     for(int j = 0; j < 128; j ++)
     {
       descriptor.at<double>(i, j) = trainingDataVector.at(i).at(j) ;
    }
  }
  } catch (bad_alloc e) {
    cerr << "memory allocate error" << endl;
    exit(1);
  }
  ifs.close();
}


// void readVecData(const options Option,
// 			  vector<double> &vectortrain,
// 			  vector< vector<double> > &trainingVectorData,
// 			  vector<double> &vectorquery,
// 			  vector< vector<double> > &testVectorData)
// {
//       ifstream ifs ( Option.trainingVectorDataFile.c_str() );
//       if (!ifs)
//       {
// 	  cerr << "Cannot open " <<  Option.trainingVectorDataFile << endl;
// 	  exit(1);
//       }
//       int flag=0;
//       while (ifs.good())
//       {
// 	  vector<double> vec;
// 	  for(unsigned int j = 0; j < 256; j++)
// 	  {
// 	      string kk;
// 	      double k;
// 	      ifs >> kk;	      
// 	      if(kk =="-nan")
// 	      {
// 		kk = "10";
// 	      }
// 	      k = atof(kk.c_str());
// 	      if (!ifs.good()) {
// 	      if (j == 0) break;
// 		cerr << "Error while reading " <<endl;
// 		exit(1);
// 	      }
// 	      vec.push_back(k);
// 	      vectortrain.push_back(k);
// 	  } // for j
// 	  if (!ifs.good()) 
// 	      break;  
// 	  trainingVectorData.push_back(vec);
// 	  flag++;
//       }
//       //cout << flag << endl;
//       ifs.close();
//       
//   
//      ifstream ifs1 ( Option.testVectorDataFile.c_str() );
//      if (!ifs1) 
//      {
// 	  cerr << "Cannot open " <<  Option.testVectorDataFile << endl;
// 	  exit(1);
//       }
//       while (ifs1.good()) 
//       {
// 	 if (!ifs1.good())
// 		break;  
// 	  vector<double> vec1;
// 	  for(unsigned int j = 0; j < 256; j++) 
// 	  {
// 		double k;
// 		ifs1 >> k;
// 		if (!ifs1.good()) {
// 		if (j == 0) break;
// 		  cerr << "Error while reading " <<endl;
// 		  exit(1);
// 		}
// 		vec1.push_back(k);
// 		vectorquery.push_back(k);
// 	  } // for j
// 	 
// 	  testVectorData.push_back(vec1);
//       }	
//       ifs1.close();
// }


void vector_distance(const vector<double> &vectorData1,
			      const vector<double> &vectorData2,
			      double &dis)
{
      
      for(unsigned int i = 0; i < vectorData1.size(); i++)
      {
	    dis += (vectorData1.at(i) - vectorData2.at(i))*(vectorData1.at(i) - vectorData2.at(i));	 
      }
}




void readMatching3Dto2D(string correspondings3D2D,vector<matchedPoints> &matchedPoints3Dto2D)
{
	
	matchedPoints matchedPoint;
	ifstream ifs ( correspondings3D2D.c_str() );
		if (!ifs) {
		  cerr << "Cannot open " << correspondings3D2D << endl;
		  exit(1);
		}
		  while (ifs.good()) {
		    if (!ifs.good()) break;
		    vector<int> feature;
		    for(unsigned int j = 0; j < 5; j++) {
		      double k;
		      ifs >> k;
		      if (j == 0)
		      {
			  matchedPoint.x3d = k;		      
		      } 
		      if (j == 1)
		      {
			  matchedPoint.y3d = k;		      
		      } 
		      if (j == 2)
		      {
			  matchedPoint.z3d = k;		      
		      } 
		      if (j == 3)
		      {
			  matchedPoint.x2d = k;		      
		      } 
		      if (j == 4)
		      {
			  matchedPoint.y2d = k;	
			  matchedPoints3Dto2D.push_back(matchedPoint);
		      } 
		  } 
		}
		ifs.close();
}

