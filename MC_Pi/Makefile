EXENAMECPU = cpu 
EXENAMEGPU = cu 
FILECPU = pi_cpu.cpp
FILEGPU = pi.cu
CXX = g++
CUXX = nvcc
FLAGS = -Wall -lm
CUFLAGS = -ccbin g++ -I../../../Common -m64 -g --thread 0 -std=c++11 -gencode arch=compute_50,code=sm_50
#--ccbin:to set the host compiler as g++ (default)
#-I include the path of any cuda related library
#-m64 specific 64 bit (default)
#-g to generate and embed debug information
#--thread 0 :Specify the maximum number of threads to be used to execute the compilation steps in parallel. This option can be used to improve the compilation speed when compiling for multiple architectures. The compiler creates number threads to execute the compilation steps in parallel. If number is 1, this option is ignored. If number is 0, the number of threads used is the number of CPUs on the machine. (default)
#-gencode arch=compute_XX, code=sm_XX where XX is the compute capability for the GPU you are using, like for GTX 960M based on Maxwell arch is SM5.0 so XX= 50

$(EXENAMECPU) : $(FILECPU)
	$(CXX) $(FLAGS) $(FILECPU) -o $(EXENAMECPU) 
$(EXENAMEGPU): $(FILEGPU)
	$(CUXX) $(CUFLAGS) $(FILEGPU) -o $(EXENAMEGPU)

clean : 
	-rm -f *.o $(EXENAMEGPU) $(EXENAMECPU)

