CC = icc
OBJECTS = read_system.o read_parameters.o build_neighbour_list.o main.o
DEBUG = 
all: $(OBJECTS)
	$(CC) $(DEBUG) -o a.out $(OBJECTS)

%.o: %.c
	$(CC) $(DEBUG) -c $<

clean:
	rm -rf *.o *.out

.phony: all
