CC = icpc
BOOST_ROOT = /home/aurora/Softwares/boost_1_70_0/install/include/
OBJECTS_C = Utilities.cpp read_system.cpp read_parameters.cpp extend_max_atoms.cpp build_neighbour_list.cpp count_types.cpp convert_coord.cpp save_to_file.cpp main.cpp
OBJECTS = Utilities.o read_system.o read_parameters.o extend_max_atoms.o build_neighbour_list.o count_types.o convert_coord.o save_to_file.o main.o
DEBUG = -O0 -g -Wall
CFLAGS = -fopenmp -I$(BOOST_ROOT)
LD_DEBUG = $(DEBUG)
LD_CFLAGS = $(CFLAGS)
LIB = -fPIC -shared -o libNNMD.so
EXE = a.out

all: exec lib

exec: $(OBJECTS)
	$(CC) $(LD_CFLAGS) $(LD_DEBUG) -o $(EXE) $(OBJECTS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEBUG) -c $<

lib:
	$(CC) $(CFLAGS) $(DEBUG) $(OBJECTS_C) $(LIB)


clean:
	rm -rf *.o *.out *.so

.phony: all
