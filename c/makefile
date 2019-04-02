CC = icc
OBJECTS = Utilities.o read_system.o read_parameters.o extend_max_atoms.o build_neighbour_list.o count_types.o convert_coord.o save_to_file.o main.o
DEBUG = -g
CFLAGS = -fopenmp
LD_DEBUG = $(DEBUG)
LD_CFLAGS = $(CFLAGS)
LIB = -fPIC -shared -o libNNMD.so
all: $(OBJECTS)
	$(CC) $(LD_CFLAGS) $(LD_DEBUG) -o a.out $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) $(DEBUG) -c $<

lib:
	$(CC) $(CFLAGS) $(DEBUG) *.c $(LIB)


clean:
	rm -rf *.o *.out *.so

.phony: all
