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
#include "cv.h"
#include "highgui.h"
using namespace cv;
#include <iostream>
#include <cstdio>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <ctime>

#include </usr/include/opencv/cv.h>
#include "opencv2/contrib/contrib.hpp"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "featurematching.h"


using namespace boost::filesystem;
using namespace std;

//ランダムに選ぶ点の数
#define Npoints 8
#define th_reprojection_error 10.0000
#define th_inlire_rate 75.0
#define th_scal 1.0




Mat pmat_ransac(vector<matchedPoints> matchedPoints3Dto2D);