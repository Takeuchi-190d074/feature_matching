cmake_minimum_required(VERSION 3.0)
project(feature_matching VERSION 1.1.0 LANGUAGES CXX)
# ビルドフラグの設定
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
# インクルードパスの追加
if(WIN32)
  include_directories(C:/local/include)
  link_directories(C:/local/lib)
else()
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()
# ビルドに関する設定
set(EXECUTABLE "feature_matching")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) #インクルードディレクトリ指定 
# add_executable(${EXECUTABLE} src/main.cpp src/${EXECUTABLE}.cpp)
add_executable(${EXECUTABLE} src/sof.cpp src/${EXECUTABLE}.cpp src/absdiff.cpp)
target_link_libraries(${EXECUTABLE} ${OpenCV_LIBS})