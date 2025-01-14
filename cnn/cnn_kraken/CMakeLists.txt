cmake_minimum_required(VERSION 3.15)
project(websocket_handler)

# set the c++ standard
set(CMAKE_CXX_STANDARD 17)

# find required packages
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenSSL REQUIRED)

# include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OPENSSL_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

# add the pybind11 module
pybind11_add_module(
    websocket_handler       # name of the output python module
    # bindings.cpp             python bindings source file
    websocket_handler.cpp   # websocket handling source file
)

# link openssl libraries AFTER the target is created
target_link_libraries(websocket_handler PRIVATE ${OPENSSL_LIBRARIES})
