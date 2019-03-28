CC = icc
OBJECTS = read_system.o main.o
DEBUG = 
all: $(OBJECTS)
	$(CC) $(DEBUG) -o a.out $(OBJECTS)

main.o: main.c
	$(CC) $(DEBUG) -c main.c 

read_system.o: read_system.c
	$(CC) $(DEBUG) -c read_system.c 

clean:
	rm -rf *.o *.out

.phony: all
