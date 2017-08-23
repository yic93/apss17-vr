CFLAGS=-std=c99 -O3
LDFLAGS=-lm -lOpenCL

all: seq opencl

seq: main.o vr_seq.o
	$(CC) $^ -o vr_seq $(CFLAGS) $(LDFLAGS)

opencl: main.o vr_opencl.o
	$(CC) $^ -o vr_opencl $(CFLAGS) $(LDFLAGS)

clean:
	rm -rf *.o vr_seq vr_opencl
