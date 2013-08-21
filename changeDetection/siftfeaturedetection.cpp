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

#include "siftfeaturedetection.h"


SIFTFeatureDetection::SIFTFeatureDetection ()
{
    cout<<"\n<===========================SIFTFeatureDetection Begin=============================>\n"<<endl; 
}

SIFTFeatureDetection::~SIFTFeatureDetection ()
{
    cout<<"\n<===========================SIFTFeatureDetection End=============================>\n"<<endl; 
}


int SIFTFeatureDetection::CallSiftGPU (Mat colorImage, int &FeatureNum, vector<SiftGPU::SiftKeypoint> &keypoint,vector<unsigned int> &dDescriptors)
{ 
    if(colorImage.empty())
	return -1;
    unsigned char *data=colorImage.ptr();
    //SiftGPU sift;					//create a SiftGPU instance
    //char * argv[] ={"-fo","-1","-v","1"};		//processing parameters first
						//-fo -1, starting from -1 octave
						//-v 1, only print out # feature and overall time
     //sift.ParseParam(4, argv);			//create an OpenGL context for computation
 
    int support = sift.CreateContextGL();	//call VerfifyContexGL instead if using your own GL context
						//int support = sift.VerifyContextGL();
    if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED)	
	return 0;
    //if(sift.RunSIFT("1.jpg"))    
    sift.RunSIFT(colorImage.size().width, colorImage.size().height, data, GL_RGB, GL_UNSIGNED_BYTE);
    //sift.SaveSIFT("1.sift");
    FeatureNum=sift.GetFeatureNum();
    vector<float> descriptors(128*FeatureNum);
    vector<SiftGPU::SiftKeypoint> keys(FeatureNum);
  
  
    //extract the features
    sift.GetFeatureVector(&keys[0], &descriptors[0]);

    /*for(int i=0;i<keys.size();i++)
    {
      //cout<<keys[i].x<<"\t"<<keys.at(i).y<<endl;
      for(int j=0;j<keys.size();j++)
      {
	if(i!=j)
	{
	  if((keys.at(i).x==keys.at(j).x)&&(keys.at(i).y==keys.at(j).y))
	  {
	    keys.erase(keys.begin()+j);
	    for(int p=0;p<128;p++)
	    {
		descriptors.erase(descriptors.begin()+j*128+p);
	    }	    
	  }
	}
      }
    }*/
    keypoint=keys;
   
    //change the type of descriptors from float to integer
    //vector<unsigned int> dDescriptors(128*FeatureNum);  
    vector<unsigned int> dDDescriptors(descriptors.size());
    dDescriptors=dDDescriptors;
    for(unsigned int i=0; i<descriptors.size(); i++)
    {
	dDescriptors[i] = (unsigned int)(0.5+512.0f*descriptors[i]);
    }     
    
    return 1;
}
