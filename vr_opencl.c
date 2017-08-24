#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>

#include "vr.h"

#define PROFILING
//#define DEBUG

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue[2];
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
    queue[0] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    queue[1] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);
#else
    queue[0] = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    queue[1] = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
#endif

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
    clReleaseCommandQueue(queue[1]);
    clReleaseCommandQueue(queue[0]);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
}

void recoverVideo(unsigned char *videoR, unsigned char *videoG, unsigned char *videoB, int *vrIdx, int N, int H, int W) {
    printf("\n");
    // create buffer
    cl_mem memR, memG, memB, memFrameMat[2];
    float **diffFrameMat = (float**) malloc(sizeof(float*) * 2);

#ifdef PROFILING
    cl_event run_event[2], read_event[2], write_event[2][3];
    unsigned long cpu_start, cpu_end;
#endif
    memR = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * H * W * N, NULL, &err);
    CHECK_ERROR(err);

    memG = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * H * W * N, NULL, &err);
    CHECK_ERROR(err);

    memB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * H * W * N, NULL, &err);
    CHECK_ERROR(err);

    int temp[2] = {0, 1};
    for (int i = 0; i < 2; i++) {
        diffFrameMat[i] = (float*) malloc(sizeof(float) * N * N * 60 * 60);

        memFrameMat[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 60 * 60 * N * N, NULL, &err);
        CHECK_ERROR(err);

    // enqueue write buffer
#ifdef PROFILING
        if (i == 0) {
            err = clEnqueueWriteBuffer(queue[i], memR, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoR, 0, NULL, &write_event[i][0]);
            CHECK_ERROR(err);
            err = clEnqueueWriteBuffer(queue[i], memG, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoG, 0, NULL, &write_event[i][1]);
            CHECK_ERROR(err);
            err = clEnqueueWriteBuffer(queue[i], memB, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoB, 0, NULL, &write_event[i][2]);
            CHECK_ERROR(err);
        } else {
            err = clEnqueueWriteBuffer(queue[i], memR, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoR, 1, &write_event[i - 1][0], &write_event[i][0]);
            CHECK_ERROR(err);
            err = clEnqueueWriteBuffer(queue[i], memG, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoG, 1, &write_event[i - 1][0], &write_event[i][1]);
            CHECK_ERROR(err);
            err = clEnqueueWriteBuffer(queue[i], memB, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoB, 1, &write_event[i - 1][0], &write_event[i][2]);
            CHECK_ERROR(err);
        }
#else
        err = clEnqueueWriteBuffer(queue[i], memR, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoR, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue[i], memG, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoG, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clEnqueueWriteBuffer(queue[i], memB, CL_FALSE, 0, sizeof(unsigned char) * H * W * N, videoB, 0, NULL, NULL);
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

        err = clSetKernelArg(kernel, 6, sizeof(cl_float) * 32 * 18, NULL);
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &memFrameMat[i]);
        CHECK_ERROR(err);

        err = clSetKernelArg(kernel, 8, sizeof(cl_int), &temp[i]);
        CHECK_ERROR(err);

        size_t global_size[2];
        if (i == 0) {
            global_size[0] = W * N;
            global_size[1] = H * N / 2;
        } else {
            global_size[0] = W * N / 2;
            global_size[1] = H * N;
        }
        size_t local_size[2] = {32, 18};

#ifdef PROFILING
        printf("[%d, workset] global: (%d, %d) / local: (%d, %d)\n", i, global_size[0], global_size[1], local_size[0], local_size[1]);
        err = clEnqueueNDRangeKernel(queue[i], kernel, 2, NULL, global_size, local_size, 0, NULL, &run_event[i]);
        CHECK_ERROR(err);
#else
        err = clEnqueueNDRangeKernel(queue[i], kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
        CHECK_ERROR(err);
#endif
        err = clFlush(queue[i]);
        CHECK_ERROR(err);

    // enqueue read buffer
#ifdef PROFILING
        err = clEnqueueReadBuffer(queue[i], memFrameMat[i], CL_FALSE, 0, sizeof(float) * 60 * 60 * N * N , diffFrameMat[i], 0, NULL, &read_event[i]);
        CHECK_ERROR(err);
#else
        err = clEnqueueReadBuffer(queue[i], memFrameMat[i[, CL_FALSE, 0, sizeof(float) * 60 * 60 * N * N , diffFrameMat[i], 0, NULL, NULL);
        CHECK_ERROR(err);
#endif

        err = clFlush(queue[i]);
        CHECK_ERROR(err);
    }

    err = clFinish(queue[0]);
    CHECK_ERROR(err);

    err = clFinish(queue[1]);
    CHECK_ERROR(err);

#ifdef PROFILING
    cpu_start = get_time();
#endif

    float *diffMat = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < 60; k++) {
                for (int l = 0; l < 60; l++) {
                    int index = ((i * 60 + k) * 60 * N) + (60 * j + l);
                    sum += diffFrameMat[0][index];
                }
            }
            diffMat[j * N + i] = sum;
            diffMat[i * N + j] = sum;
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
    for (int i = 0; i < 2; i++) {
        cl_ulong write_start, write_end;
        cl_ulong run_start, run_end;
        cl_ulong read_start, read_end;

        printf("[%d]\n", i);

        clGetEventProfilingInfo(write_event[i][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_start, NULL);
        clGetEventProfilingInfo(write_event[i][2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_end, NULL);
        printf("[write] %lu ns\n", (write_end - write_start));

        clGetEventProfilingInfo(run_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &run_start, NULL);
        clGetEventProfilingInfo(run_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &run_end, NULL);
        printf("[run] %lu ns\n", (run_end - run_start));

        clGetEventProfilingInfo(read_event[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_start, NULL);
        clGetEventProfilingInfo(read_event[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_end, NULL);
        printf("[read] %lu ns\n", (read_end - read_start));
    }

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
