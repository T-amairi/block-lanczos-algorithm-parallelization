all:
	cd sequential; make
	cd openMP; make
	cd mpi; make
	cd hybrid; make

clean:
	cd sequential; make clean
	cd openMP; make clean
	cd mpi; make clean
	cd hybrid; make clean