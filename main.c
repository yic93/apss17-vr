#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "vr.h"

const int H = 1080, W = 1920, C = 3;
int N;
unsigned char *videoR, *videoG, *videoB;
int *vrIdx;

// get current epoch in seconds
double getTime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1.0e-6*tv.tv_usec;
}

// read video binary file
void readBin(char *fn) {
    FILE *fin = fopen(fn, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open %s\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("Reading %s...", fn);
    fflush(stdout);
    long sz;
    fseek(fin, 0, SEEK_END);
    sz = ftell(fin);
    N = sz / (H * W * C);
    fseek(fin, 0, SEEK_SET);
    unsigned char *video = (unsigned char*)malloc(sz);
    fread(video, 1, sz, fin);
    fclose(fin);
    videoR = (unsigned char*)malloc(sz / 3);
    videoG = (unsigned char*)malloc(sz / 3);
    videoB = (unsigned char*)malloc(sz / 3);
    unsigned char *p = video;
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                videoR[(n * H + h) * W + w] = *p++;
                videoG[(n * H + h) * W + w] = *p++;
                videoB[(n * H + h) * W + w] = *p++;
            }
        }
    }
    free(video);
    printf(" Done.\n");
}

// write ordered frame index
void writeIdx(char *fn) {
    FILE *fout = fopen(fn, "w");
    for (int i = 0; i < N; ++i) {
        fprintf(fout, "%d\n", vrIdx[i]);
    }
    fclose(fout);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input bin> <output out>\n", argv[0]);
        fprintf(stderr, " e.g., %s data/video0.bin result0.out\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    readBin(argv[1]);
    vrIdx = (int*)malloc(N * sizeof(int));

    init();

    printf("Recovering video...");
    fflush(stdout);
    double st = getTime();
    recoverVideo(videoR, videoG, videoB, vrIdx, N, H, W);
    double et = getTime();
    printf(" Done.\n");
    printf("Elapsed time : %fs\n", et - st);

    writeIdx(argv[2]);

    free(videoR);
    free(videoG);
    free(videoB);
    free(vrIdx);

    return 0;
}
