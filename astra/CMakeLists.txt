cmake_minimum_required(VERSION 3.10)
project(astra_capture)

set(CMAKE_CXX_STANDARD 17)

find_library(OBBEC_SDK OrbbecSDK /usr/local/lib NO_DEFAULT_PATH)
if(NOT OBBEC_SDK)
    find_library(OBBEC_SDK OrbbecSDK)
endif()

add_executable(astra_capture capture.cpp)
target_include_directories(astra_capture PRIVATE /usr/local/include)
target_link_libraries(astra_capture ${OBBEC_SDK})
