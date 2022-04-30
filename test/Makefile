CC = mpicc
CFLAGS = -std=c99 -g -Wall -Wextra -Werror -O3 -march=native -I.
LDFLAGS =

all: test checker_modp
test: mmio.o test.o
test.o: test.c mmio.h
checker_modp:   mmio.o checker_modp.o
checker_modp.o: checker_modp.c mmio.h

clean:
	rm -f *.o
	rm -f test checker_modp
