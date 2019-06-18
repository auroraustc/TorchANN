#define CC and CXX
CC = g++
CXX = g++
#define the BOOST C++ library root and include path
BOOST_ROOT = /home/aurora/Softwares/boost_1_70_0/install/include/
BOOST_INC = $(BOOST_ROOT)/include/
#Objects used for make
OBJECTS_CXX = Utilities.cpp read_system.cpp read_parameters.cpp extend_max_atoms.cpp build_neighbour_list.cpp count_types.cpp convert_coord.cpp save_to_file.cpp main.cpp
OBJECTS = Utilities.o read_system.o read_parameters.o extend_max_atoms.o build_neighbour_list.o count_types.o convert_coord.o save_to_file.o main.o
OBJECTS_P = Utilities.o read_system.o read_parameters.o extend_max_atoms.o build_neighbour_list.o count_types.o convert_coord.o save_to_file.o main_partial.o
#flags
DEBUG = -O0 -g -Wall
CXXFLAGS = -fopenmp -std=c++11 -I$(BOOST_ROOT)
LD_DEBUG = $(DEBUG)
LD_CXXFLAGS = $(CXXFLAGS)
LIB = -fPIC -shared -o libANN.so
EXE = GENERATE
EXE_P = GENERATE_P

all: exec lib

partial: exec_partial

exec: $(OBJECTS)
	$(CXX) $(LD_CXXFLAGS) $(LD_DEBUG) -o $(EXE) $(OBJECTS)

exec_partial: $(OBJECTS_P)
	$(CXX) $(LD_CXXFLAGS) $(LD_DEBUG) -o $(EXE_P) $(OBJECTS_P)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEBUG) -c $<

lib:
	$(CXX) $(CXXFLAGS) $(DEBUG) $(OBJECTS_CXX) $(LIB)

clean:
	rm -rf *.o *.out *.so GEN*

.phony: all
