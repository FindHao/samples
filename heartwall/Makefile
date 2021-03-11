
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif
CUDA_DIR ?= /usr/local/cuda

# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o  
	nvcc main.o ./AVI/avilib.o ./AVI/avimod.o -I/usr/local/cuda/include -lm -o heartwall -O3 -g -lineinfo -arch sm_75 -L$(CUDA_LIB_DIR) 

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3 -g -lineinfo -arch sm_75 -L$(CUDA_LIB_DIR) 

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
