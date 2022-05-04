EXENAME = pi

CXX = g++
CUXX = nvcc
FLAGS = -Wall -lm

picpu : pi_cpu.cpp
	$(CXX) $(FLAGS) pi_cpu.cpp -o picpu
picuda : pi.cu
	$(CUXX)  pi.cu -o picuda

clean : 
	-rm -f *.o picpu picu

