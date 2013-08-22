cmake_minimum_required(VERSION 2.8)
SET(CMAKE_BUILD_TYPE Release)
FIND_PACKAGE(OpenCV REQUIRED)
ADD_EXECUTABLE(changedetection changedetection.cpp fileoperation.cpp featurematching.cpp pmat_ransac.cpp readFeatures.cpp siftfeaturedetection.cpp main.cpp rply.c poseestimation.cpp)
TARGET_LINK_LIBRARIES(changedetection ${OpenCV_LIBS} siftgpu boost_system boost_program_options boost_filesystem ANN)

