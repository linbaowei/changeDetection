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

#include "common.h"

int main(int argc, char **argv)
{    
	option = parseOptions(argc, argv); 
	cout << "\n\n\nstart calibration...";
	Mat intrinsic, distCoeffs;
	poseEstimation poseestimation;
	poseestimation.cameraCalibration(intrinsic, distCoeffs);
// 	cout  << intrinsic << endl;
// 	cout  << distCoeffs << endl;
	cout << "............................" << endl;
	cout << "calibration finished.\n\n\n" << endl;
	methodopencvPoseEstimation(intrinsic, distCoeffs);
	cout << "\n\nFINISH." << endl;
	return 1;
}
    

/**
* Parse command line options.
* return options struct
*/
options parseOptions(int argc, char** argv)
{      
	namespace po = boost::program_options;      
	po::options_description desc("Options");
	desc.add_options()  
	("help", "This help message.")
	("plyFileof3DpointCloudFile", po::value<string>(), "plyFileof3DpointCloudFile")
	("3DKeypointsFile", po::value<string>(), "3DKeypointsFile")
	//("all3DpointsBefroeThresholding_file", po::value<string>(), "all3DpointsBefroeThresholding_file")
	("trainingImagesKeysPath", po::value<string>(),"trainingImagesKeysPath")
	("testImagesPath", po::value<string>(), "testImagesPath")
	("topNearestImagesNum", po::value<int>(), "topNearestImagesNum");
	
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    
	
	options Opt;
	
	if (vm.count("help"))
	{
	cout << desc << endl;
	exit(0);
	}
	int flag = 0;
	if (vm.count("plyFileof3DpointCloudFile"))        	
		Opt.plyFileof3DpointCloudFile_train = vm["plyFileof3DpointCloudFile"].as<string>();
	else{cout << "plyFileof3DpointCloudFile  was not set yet." << endl; flag++;}
	
	if (vm.count("3DKeypointsFile"))       
		Opt.ThreeDKeypointsFile = vm["3DKeypointsFile"].as<string>();
	else{cout << "3DKeypointsFile was not set yet." << endl; flag++;}
	
	if (vm.count("trainingImagesKeysPath")) 		
		Opt.trainingImagesKeysPath = vm["trainingImagesKeysPath"].as<string>();
	else{cout << "trainingImagesKeysPath was not set yet." << endl; flag++;}
	
	if (vm.count("testImagesPath"))  	
		Opt.testImagesPath = vm["testImagesPath"].as<string>();
	else{cout << "testImagesPath was not set yet." << endl; flag++;}
	
	if (vm.count("topNearestImagesNum"))  	
		Opt.topNearestImagesNum = vm["topNearestImagesNum"].as<int>();
	else{cout << "topNearestImagesNum was not set yet." << endl; flag++;}
	
	if(flag >0)
	{		
		    cout << "\nusage:\n" << argv[0] << "\n\t--plyFileof3DpointCloudFile" << "\t\t[YOURINPUT]\n"
		    << "\t--3DKeypointsFile" << "\t\t\t[YOURINPUT]\n"
		    << "\t--trainingImagesKeysPath" << "\t\t[YOURINPUT]\n"
		    << "\t--testImagesPath" << "\t\t\t[YOURINPUT]\n"
		    << "\t--topNearestImagesNum" << "\t\t\t[YOURINPUT]\n" << endl;
		    exit(1);  
	}	
	
	cout << "plyFileof3DpointCloudFile "		<< Opt.plyFileof3DpointCloudFile_train << endl;
	cout << "3DKeypointsFile "				<< Opt.ThreeDKeypointsFile	<< endl;
	cout << "trainingImagesKeysPath " 		<< Opt.trainingImagesKeysPath 		<< endl;
	cout << "testImagesPath "				<< Opt.testImagesPath 			<< endl;
	cout << "topNearestImagesNum "		<< Opt.topNearestImagesNum 			<< endl;
	return Opt;
}
    
    
    
    
    
    