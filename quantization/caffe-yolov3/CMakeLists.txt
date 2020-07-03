# 指定cmake版本
cmake_minimum_required(VERSION 2.8)
# 指定项目名称，一般和项目的文件夹名称对应
project(sysDetectSpeed)
# 添加c++11标准支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

# setup CUDA
# 找到后面需要库和头文件的包
find_package(CUDA)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_30,code=sm_30 
		-gencode arch=compute_35,code=sm_35 
		-gencode arch=compute_50,code=sm_50 
		-gencode arch=compute_52,code=sm_52 
		-gencode arch=compute_60,code=sm_60 
		-gencode arch=compute_61,code=sm_61 
		-gencode arch=compute_61,code=compute_61
)

# setup project output paths
# 设置路径（下面生成共享库的路径），即生成的共享库在工程文件夹下的lib文件夹
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}) #/build/x86_64
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include) #/build/x86_64/include

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR}) #/build/x86_64/include
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin) #/build/x86_64/bin

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}") #x86_64
message("-- output path:  ${PROJECT_OUTPUT_DIR}") #/build/x86_64
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin) #/build/x86_64/bin
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)#/build/x86_64/lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)#/build/x86_64/lib


# build C/C++ interface
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(${PROJECT_INCLUDE_DIR}
#	/home/nvidia/caffe/include
	/home/user/Work/mjw/quantization/ezai/caffe_base_ezai/include
#	/home/nvidia/caffe/build/include
	/home/user/Work/mjw/quantization/ezai/caffe_base_ezai/build/include
)


file(GLOB inferenceSources *.cpp *.cu )
file(GLOB inferenceIncludes *.h )

cuda_add_library(sysDetectSpeed SHARED ${inferenceSources})
target_link_libraries(sysDetectSpeed 
#	/home/nvidia/caffe/build/lib/libcaffe.so  
	/home/user/Work/mjw/quantization/ezai/caffe_base_ezai/build/lib/libcaffe.so
	/usr/lib/x86_64-linux-gnu/libglog.so  
	/usr/lib/x86_64-linux-gnu/libgflags.so.2
    /usr/lib/x86_64-linux-gnu/libboost_system.so  
	/usr/lib/x86_64-linux-gnu/libGLEW.so.1.13  
)


# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()


# create symbolic link for network data
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/data/networks" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/networks" )
  
add_subdirectory(detectnet)


#setup opencv
FIND_PACKAGE(OpenCV REQUIRED)

TARGET_LINK_LIBRARIES(sysDetectSpeed ${OpenCV_LIBS})

# install
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/sysDetectSpeed)
endforeach()

# install the shared library
install(TARGETS sysDetectSpeed DESTINATION lib/sysDetectSpeed EXPORT sysDetectSpeedConfig)

# install the cmake project, for importing
install(EXPORT sysDetectSpeedConfig DESTINATION share/sysDetectSpeed/cmake)
