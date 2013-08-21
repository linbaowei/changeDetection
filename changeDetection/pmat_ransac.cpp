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


#include "pmat_ransac.h"

struct corresponding_points{
	double x_3d;
	double y_3d;
	double z_3d;
	double x_2d;
	double y_2d;
};
double distance_2d(const double x1, const double y1, const double x2, const double y2){

	double dis;
	dis = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
	dis = sqrt(dis);

	return dis;
}

Mat pmat_ransac(vector<matchedPoints> matchedPoints3Dto2D){

	srand( (unsigned)time( NULL ));	//乱数seed
	int corres_num_MAX = matchedPoints3Dto2D.size();	//対応点の組の総数
	Mat P = Mat::zeros(3,4,CV_64F);
	if (corres_num_MAX <= Npoints)
	  return P;
	//入力ファイル（X,Y,Z,x,y）の読み込み
	struct corresponding_points points[matchedPoints3Dto2D.size()];		//points[0]がinputFileの1行目の点
	for(unsigned int i =0; i < matchedPoints3Dto2D.size(); i++)
	{
	      points[i].x_3d = matchedPoints3Dto2D.at(i).x3d;
	      points[i].y_3d = matchedPoints3Dto2D.at(i).y3d;
	      points[i].z_3d = matchedPoints3Dto2D.at(i).z3d;
	      points[i].x_2d = matchedPoints3Dto2D.at(i).x2d;
	      points[i].y_2d = matchedPoints3Dto2D.at(i).y2d;
	}
	double flag_reprojection_error = 0;
	CvMat *matrixP = cvCreateMat(3, 4, CV_64F);
	int selected[Npoints];

	double det_R = 0;	
	int loop_count = 0;	
	double time = 0;
	while(true){
	  TickMeter tm;
	  tm.start();
		//cout <<  (flag_reprojection_error / (corres_num_MAX - Npoints)) * 100  << endl;
			set<int> select;	//選んだ点の番号を格納（0~corres_num_MAX-1）
			int r;
			
			for(int i = 0; i < Npoints; ++i){
				do{
					r = rand() % corres_num_MAX;
				} while(select.find(r) != select.end());
				select.insert(r);
				selected[i] = r;
			}
			
			//行列A（Ap＝0）
			CvMat *matrixA = cvCreateMat(Npoints * 2, 12, CV_64F);	//行列A(Npoints,12)
			CvMat *matrix3DX = cvCreateMat(4, Npoints, CV_64F);	//3D点を格納(4, Npoints)
			CvMat *matrix2DX = cvCreateMat(3, Npoints, CV_64F);	//2D点を格納(3, Npoints)
			int mat_low = 0;
			int mat_low2 = 0;
			set<int>::iterator it = select.begin();
			while(it != select.end()){
				//行列Aに値を代入	A(N,0)~A(N,11)
				cvmSet(matrixA, mat_low, 0, 0.0);
				cvmSet(matrixA, mat_low, 1, 0.0);
				cvmSet(matrixA, mat_low, 2, 0.0);
				cvmSet(matrixA, mat_low, 3, 0.0);
				cvmSet(matrixA, mat_low, 4, points[*it].x_3d);
				cvmSet(matrixA, mat_low, 5, points[*it].y_3d);
				cvmSet(matrixA, mat_low, 6, points[*it].z_3d);
				cvmSet(matrixA, mat_low, 7, 1.0);
				cvmSet(matrixA, mat_low, 8, - points[*it].y_2d * points[*it].x_3d);
				cvmSet(matrixA, mat_low, 9, - points[*it].y_2d * points[*it].y_3d);
				cvmSet(matrixA, mat_low, 10, - points[*it].y_2d * points[*it].z_3d);
				cvmSet(matrixA, mat_low, 11, - points[*it].y_2d);
				//行列Aに値を代入	A(N+1,0)~A(N+1,11)
				cvmSet(matrixA, mat_low + 1, 0, points[*it].x_3d);
				cvmSet(matrixA, mat_low + 1, 1, points[*it].y_3d);
				cvmSet(matrixA, mat_low + 1, 2, points[*it].z_3d);
				cvmSet(matrixA, mat_low + 1, 3, 1.0);
				cvmSet(matrixA, mat_low + 1, 4, 0.0);
				cvmSet(matrixA, mat_low + 1, 5, 0.0);
				cvmSet(matrixA, mat_low + 1, 6, 0.0);
				cvmSet(matrixA, mat_low + 1, 7, 0.0);
				cvmSet(matrixA, mat_low + 1, 8, - points[*it].x_2d * points[*it].x_3d);
				cvmSet(matrixA, mat_low + 1, 9, - points[*it].x_2d * points[*it].y_3d);
				cvmSet(matrixA, mat_low + 1, 10, - points[*it].x_2d * points[*it].z_3d);
				cvmSet(matrixA, mat_low + 1, 11, - points[*it].x_2d);
				//行列3DX:選ばれた3D点を格納
				cvmSet(matrix3DX, 0, mat_low2, points[*it].x_3d);
				cvmSet(matrix3DX, 1, mat_low2, points[*it].y_3d);
				cvmSet(matrix3DX, 2, mat_low2, points[*it].z_3d);
				cvmSet(matrix3DX, 3, mat_low2, 1.0);
				//行列2DX:選ばれた2D点を格納
				cvmSet(matrix2DX, 0, mat_low2, points[*it].x_2d);
				cvmSet(matrix2DX, 1, mat_low2, points[*it].y_2d);
				cvmSet(matrix2DX, 2, mat_low2, 1.0);
				++it;
				mat_low = mat_low + 2;
				mat_low2++;
			}

			CvMat *matrix3DX2 = cvCreateMat(4, corres_num_MAX - Npoints, CV_64F);
			CvMat *matrix2DX2 = cvCreateMat(3, corres_num_MAX - Npoints, CV_64F);

			mat_low2 = 0;
			for(int i = 0; i < corres_num_MAX - Npoints; i++){
				int f = 0;
				it = select.begin();
				while(it != select.end()){
					if(*it == i) f = 1;
					++it;
				}

				if(f == 0){
					cvmSet(matrix3DX2, 0, mat_low2, points[i].x_3d);
					cvmSet(matrix3DX2, 1, mat_low2, points[i].y_3d);
					cvmSet(matrix3DX2, 2, mat_low2, points[i].z_3d);
					cvmSet(matrix3DX2, 3, mat_low2, 1.0);

					cvmSet(matrix2DX2, 0, mat_low2, points[i].x_2d);
					cvmSet(matrix2DX2, 1, mat_low2, points[i].y_2d);
					cvmSet(matrix2DX2, 2, mat_low2, 1.0);

					mat_low2++;
				}
			}

			//行列Bの計算(B=A^T A)
			CvMat *matrixB = cvCreateMat(12, 12, CV_64F);	//行列B(12,12)
			cvMulTransposed(matrixA, matrixB, 1, NULL);	//A^TとAの乗算をBに代入（3番目の引数は乗算の順番）
			cvReleaseMat(&matrixA);

			CvMat *matrixW = cvCreateMat(12, 1, CV_64F);
			CvMat *matrixU = cvCreateMat(12, 12, CV_64F);
			cvSVD(matrixB, matrixW, matrixU, NULL, 0);

			cvReleaseMat(&matrixB);
			cvReleaseMat(&matrixW);

			//
			//射影行列を使って再投影
			//

			//仮に求まったP
			cvmSet(matrixP, 0, 0, cvmGet(matrixU, 0, 11));
			cvmSet(matrixP, 0, 1, cvmGet(matrixU, 1, 11));
			cvmSet(matrixP, 0, 2, cvmGet(matrixU, 2, 11));
			cvmSet(matrixP, 0, 3, cvmGet(matrixU, 3, 11));
			cvmSet(matrixP, 1, 0, cvmGet(matrixU, 4, 11));
			cvmSet(matrixP, 1, 1, cvmGet(matrixU, 5, 11));
			cvmSet(matrixP, 1, 2, cvmGet(matrixU, 6, 11));
			cvmSet(matrixP, 1, 3, cvmGet(matrixU, 7, 11));
			cvmSet(matrixP, 2, 0, cvmGet(matrixU, 8, 11));
			cvmSet(matrixP, 2, 1, cvmGet(matrixU, 9, 11));
			cvmSet(matrixP, 2, 2, cvmGet(matrixU, 10, 11));
			cvmSet(matrixP, 2, 3, cvmGet(matrixU, 11, 11));

			cvReleaseMat(&matrixU);

			CvMat *matrix2DreX = cvCreateMat(3, Npoints, CV_64F);
			cvMatMulAdd(matrixP, matrix3DX, NULL, matrix2DreX);


			CvMat *matrix2DreX2 = cvCreateMat(3, corres_num_MAX - Npoints, CV_64F);
			cvMatMulAdd(matrixP, matrix3DX2, NULL, matrix2DreX2);

			cvReleaseMat(&matrix3DX);
			cvReleaseMat(&matrix3DX2);

			flag_reprojection_error = 0;

			for(int i = 0; i < corres_num_MAX - Npoints; i++){
				cvmSet(matrix2DreX2, 0, i, cvmGet(matrix2DreX2, 0, i) / cvmGet(matrix2DreX2, 2, i));
				cvmSet(matrix2DreX2, 1, i, cvmGet(matrix2DreX2, 1, i) / cvmGet(matrix2DreX2, 2, i));
				cvmSet(matrix2DreX2, 2, i, cvmGet(matrix2DreX2, 2, i) / cvmGet(matrix2DreX2, 2, i));

				//再投影誤差の計算
				double dis2;
				dis2 = distance_2d( cvmGet(matrix2DreX2, 0, i), cvmGet(matrix2DreX2, 1, i), 
						    cvmGet(matrix2DX2,   0, i), cvmGet(matrix2DX2,   1, i));

				if(dis2 < th_reprojection_error / th_scal)	
				  flag_reprojection_error++;
			}

			cvReleaseMat(&matrix2DX);
			cvReleaseMat(&matrix2DreX);

			cvReleaseMat(&matrix2DX2);
			cvReleaseMat(&matrix2DreX2);
			 
			//cout <<flag_reprojection_error / (corres_num_MAX - Npoints) * 100 << endl;
			if(flag_reprojection_error / (corres_num_MAX - Npoints) * 100 > th_inlire_rate)
			{
			      CvMat *matrixP_SVD = cvCreateMat(3, 3, CV_64F);
			      cvSetZero(matrixP_SVD);
			      for(int i = 0; i < 3; i++)
			      {
				      cvmSet(matrixP_SVD, i, 0, cvmGet(matrixP, i, 0) / cvmGet(matrixP, 2, 3));
				      cvmSet(matrixP_SVD, i, 1, cvmGet(matrixP, i, 1) / cvmGet(matrixP, 2, 3));
				      cvmSet(matrixP_SVD, i, 2, cvmGet(matrixP, i, 2) / cvmGet(matrixP, 2, 3));
			      }

			      CvMat *matrixW1 = cvCreateMat(3, 3, CV_64F);
			      cvSetZero(matrixW1);
			      CvMat *matrixU1 = cvCreateMat(3, 3, CV_64F);
			      cvSetZero(matrixU1);
			      CvMat *matrixV_T1 = cvCreateMat(3, 3, CV_64F);
			      cvSetZero(matrixV_T1);
			      cvSVD(matrixP_SVD, matrixW1, matrixU1, matrixV_T1, CV_SVD_V_T);
			      cvReleaseMat(&matrixP_SVD);
			      cvReleaseMat(&matrixW1);
			      CvMat *matrixR_SVD = cvCreateMat(3, 3, CV_64F);
			      cvSetZero(matrixR_SVD);
			      cvMatMulAdd(matrixU1, matrixV_T1, NULL, matrixR_SVD);			      
			      det_R = cvDet(matrixR_SVD);
			      //cout << det_R << endl;
			      
			      if(abs(det_R-1)<0.1)
			      {
				    cerr << "det_R: " << det_R << endl;		
				    cerr << "rate: " << (flag_reprojection_error / (corres_num_MAX - Npoints)) * 100 << endl;
				    Mat projectionMatrix = Mat(3, 4, CV_64FC1);
				    for(int i = 0; i < 3; i++)
					for(int j = 0; j < 4; j++)
					  projectionMatrix.at<double>(i, j) = cvmGet(matrixP, i, j) / cvmGet(matrixP, 2, 3);			    
				    cvReleaseMat(&matrixP);
				    return projectionMatrix;
			      }
			}
			tm.stop();
			time+= tm.getTimeMilli();
			if(time >100000)
			  return P;
		}//loop end	
}