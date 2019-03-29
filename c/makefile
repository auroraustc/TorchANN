CC = icc
OBJECTS = read_system.o read_parameters.o build_neighbour_list.o count_types.o main.o
DEBUG = -g
CFLAGS = -fopenmp
LD_DEBUG = $(DEBUG)
LD_CFLAGS = $(CFLAGS)
all: $(OBJECTS)
	$(CC) $(LD_CFLAGS) $(LD_DEBUG) -o a.out $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) $(DEBUG) -c $<

clean:
	rm -rf *.o *.out

.phony: all
