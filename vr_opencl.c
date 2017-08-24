#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>

#include "vr.h"

#define PROFILING
#define DEBUG

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

char *get_source_code(const char *file_name, size_t *len) {
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char *)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';

    fclose(file);

    *len = length;
    return source_code;
}

#ifdef PROFILING
unsigned long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 100000) + tv.tv_usec) * 1000;
}
#endif

void init() {
    // get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // create command queue
#ifdef PROFILING
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CHECK_ERROR(err);

    // get source code
    size_t source_size;
    char *source_code = get_source_code("kernel.cl", &source_size);
    program = clCreateProgramWithSource(context, 1, (const char**) &source_code, &source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char *) malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    // create kernel
    kernel = clCreateKernel(program, "recover_video", &err);
    CHECK_ERROR(err);
}

void release() {
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
}

void recoverVideo(unsigned char *videoR, unsigned char *videoG, unsigned char *videoB, int *vrIdx, int N, int H, int W) {
    // create buffer
    cl_mem memR, memG, memB, memFrameMat, memDiffMat;
    long rgb_size = sizeof(unsigned char) * H * W * N;
    long float_nn_size = sizeof(float) * N * N;
    float *diffFrameMat = (float*)malloc(60 * 60 * N * N * sizeof(float));

#ifdef PROFILING
    cl_ulong write_start, write_end;
    cl_ulong run_start, run_end;
    cl_ulong read_start, read_end;
    cl_event run_event, read_event, write_event[3];
    unsigned long cpu_start, cpu_end;
#endif

    memR = clCreateBuffer(context, CL_MEM_READ_ONLY, rgb_size, NULL, &err);
    CHECK_ERROR(err);

    memG = clCreateBuffer(context, CL_MEM_READ_ONLY, rgb_size, NULL, &err);
    CHECK_ERROR(err);

    memB = clCreateBuffer(context, CL_MEM_READ_ONLY, rgb_size, NULL, &err);
    CHECK_ERROR(err);

    //memDiffMat = clCreateBuffer(context, CL_MEM_READ_WRITE, float_nn_size, NULL, &err);
    //CHECK_ERROR(err);

    memFrameMat = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 60 * 60 * N  * N, NULL, &err);
    CHECK_ERROR(err);


    // enqueue write buffer
#ifdef PROFILING
    err = clEnqueueWriteBuffer(queue, memR, CL_FALSE, 0, rgb_size, videoR, 0, NULL, &write_event[0]);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, memG, CL_FALSE, 0, rgb_size, videoG, 0, NULL, &write_event[1]);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, memB, CL_FALSE, 0, rgb_size, videoB, 0, NULL, &write_event[2]);
    CHECK_ERROR(err);
#else
    err = clEnqueueWriteBuffer(queue, memR, CL_FALSE, 0, rgb_size, videoR, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, memG, CL_FALSE, 0, rgb_size, videoG, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, memB, CL_FALSE, 0, rgb_size, videoB, 0, NULL, NULL);
    CHECK_ERROR(err);
#endif

    // kernal argument setting
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memR);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memG);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memB);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &N);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 4, sizeof(cl_int), &H);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 5, sizeof(cl_int), &W);
    CHECK_ERROR(err);

    //err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &memDiffMat);
    //CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 6, sizeof(cl_float) * 32 * 18, NULL);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &memFrameMat);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 8, sizeof(cl_char) * 32 * 18 * 2, NULL);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 9, sizeof(cl_char) * 32 * 18 * 2, NULL);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 10, sizeof(cl_char) * 32 * 18 * 2, NULL);
    CHECK_ERROR(err);

    // run kernel
    size_t global_size[2] = {W * N, H * N};
    size_t local_size[2] = {32, 18};
    //size_t local_size[2] = {W, H};

    //while (global_size[0] % local_size[0] != 0) global_size[0]++;
    //while (global_size[1] % local_size[1] != 0) global_size[1]++;


#ifdef PROFILING
    printf("\n[workset] global: (%d, %d) / local: (%d, %d)\n", global_size[0], global_size[1], local_size[0], local_size[1]);

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &run_event);
    CHECK_ERROR(err);

    // enqueue read buffer
    //err = clEnqueueReadBuffer(queue, memDiffMat, CL_FALSE, 0, float_nn_size, diffMat, 0, NULL, &read_event);
    //CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, memFrameMat, CL_FALSE, 0, float_nn_size * 60 * 60, diffFrameMat, 0, NULL, &read_event);
    CHECK_ERROR(err);

#else
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    // enqueue read buffer
    //err = clEnqueueReadBuffer(queue, memDiffMat, CL_FALSE, 0, float_nn_size, diffMat, 0, NULL, NULL);
    //CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, memFrameMat, CL_FALSE, 0, float_nn_size * 60 * 60, diffFrameMat, 0, NULL, NULL);
    CHECK_ERROR(err);
#endif

    err = clFlush(queue);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);

#ifdef PROFILING
    cpu_start = get_time();
#endif

    float *diffMat = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < 60; k++) {
                for (int l = 0; l < 60; l++) {
                    int index = ((i * 60 + k) * 60 * N) + (60 * j + l);
                    sum += diffFrameMat[index];
                }
            }
            diffMat[j * N + i] = sum;
        }
    }

#ifdef DEBUG
    FILE* fp = fopen("diff_mat_cl.txt", "w");
    for (int i = 0; i < N * N; i++) {
        fprintf(fp, "[%d] %f\n", i, diffMat[i]);
    }
    fclose(fp);
#endif

    int *used = (int*)calloc(N, sizeof(int));
    vrIdx[0] = 0;
    used[0] = 1;
    for (int i = 1; i < N; ++i) {
        int f0 = vrIdx[i - 1], f1, minf = -1;
        float minDiff;
        for (f1 = 0; f1 < N; ++f1) {
            if (used[f1] == 1) continue;
            if (minf == -1 || minDiff > diffMat[f0 * N + f1]) {
                minf = f1;
                minDiff = diffMat[f0 * N + f1];
            }
        }
        vrIdx[i] = minf;
        used[minf] = 1;
    }

#ifdef PROFILING
    cpu_end = get_time();
#endif

#ifdef PROFILING
    clGetEventProfilingInfo(write_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_start, NULL);
    clGetEventProfilingInfo(write_event[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_end, NULL);
    printf("[write] %lu ns\n", (write_end - write_start));

    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &run_start, NULL);
    clGetEventProfilingInfo(run_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &run_end, NULL);
    printf("[run] %lu ns\n", (run_end - run_start));

    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_end, NULL);
    printf("[read] %lu ns\n", (read_end - read_start));

    printf("[cpu] %lu ns\n", (cpu_end - cpu_start));
#endif

    return;

    release();

    free(diffMat);
    free(used);

    clReleaseMemObject(memR);
    clReleaseMemObject(memG);
    clReleaseMemObject(memB);
}
