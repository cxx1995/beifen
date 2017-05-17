all:
	nvcc -arch=sm_60 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe
