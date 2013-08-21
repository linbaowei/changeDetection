cmake_minimum_required(VERSION 2.8)
INCLUDE_DIRECTORIES(/usr/include/opencv)
SET(CMAKE_BUILD_TYPE Release)
ADD_EXECUTABLE(changedetection changedetection.cpp fileoperation.cpp featurematching.cpp pmat_ransac.cpp readFeatures.cpp siftfeaturedetection.cpp main.cpp rply.c poseestimation.cpp)
TARGET_LINK_LIBRARIES(changedetection opencv_core opencv_imgproc opencv_nonfree opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy siftgpu boost_system boost_program_options boost_filesystem ANN)

