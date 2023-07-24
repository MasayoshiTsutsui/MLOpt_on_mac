cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(vgg11)

find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(vgg11 run_vgg11.cpp)
target_link_libraries(vgg11 "${TORCH_LIBRARIES}")
set_property(TARGET vgg11 PROPERTY CXX_STANDARD 17)
