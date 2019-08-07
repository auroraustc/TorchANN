cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(train_nompi)

find_package(Torch REQUIRED)

#Define c++ compiler here
SET(CMAKE_CXX_COMPILER "mpicxx")

add_definitions(${CXX_SRCs})
SET(CXX_SRCs "${CMAKE_SOURCE_DIR}/train_nompi.cpp ${CMAKE_SOURCE_DIR}/train_nompi_Utilities.cpp")

add_executable(train_nompi train_nompi.cpp train_nompi_Utilities.cpp)
target_link_libraries(train_nompi "${TORCH_LIBRARIES}")
set_property(TARGET train_nompi PROPERTY CXX_STANDARD 11)

#Change DEBUG_FLAGS here
add_definitions(${DEBUG_FLAGS})
SET(DEBUG_FLAGS    "-g")

#Change ADDITIONAL_FLAGS here
add_definitions(${ADDITIONAL_FLAGS})
SET(ADDITIONAL_FLAGS "-fopenmp -cxx=g++")

add_definitions(${ADDITIONAL_LIBS})
add_definitions(${libANN_PATH})
SET(libANN_PATH "${CMAKE_SOURCE_DIR}/../")
SET(ADDITIONAL_LIBS "-L${libANN_PATH} -lANN")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${DEBUG_FLAGS} ${ADDITIONAL_FLAGS} ${ADDITIONAL_LIBS}")
