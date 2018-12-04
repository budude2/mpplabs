
NVCC        = nvcc
NVCC_FLAGS  = -O3
LD_FLAGS    = -lcudart -lopencv_core -lopencv_highgui -lopencv_imgcodecs
EXE	        = avg
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu support.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
