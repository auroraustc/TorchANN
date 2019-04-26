CC = icc
OBJECTS_C = Utilities.c read_system.c read_parameters.c extend_max_atoms.c build_neighbour_list.c count_types.c convert_coord.c save_to_file.c main.c
OBJECTS = Utilities.o read_system.o read_parameters.o extend_max_atoms.o build_neighbour_list.o count_types.o convert_coord.o save_to_file.o main.o
DEBUG = -g
CFLAGS = -fopenmp
LD_DEBUG = $(DEBUG)
LD_CFLAGS = $(CFLAGS)
LIB = -fPIC -shared -o libNNMD.so

all: exec lib

exec: $(OBJECTS)
	$(CC) $(LD_CFLAGS) $(LD_DEBUG) -o a.out $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) $(DEBUG) -c $<

lib:
	$(CC) $(CFLAGS) $(DEBUG) $(OBJECTS_C) $(LIB)


clean:
	rm -rf *.o *.out *.so

.phony: all
