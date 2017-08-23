#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "vr.h"

void init() {
    // dummy
}

/*
 * videoR : R color channel of video. unsigned char [N][H][W] array, flattened to 1-dimension
 * videoG, videoB : similar to videoR
 * vrIdx : int [N] array. write recovered frame order to this array
 * N : number of frames
 * H : frame height (fixed to 1080)
 * W : frame width (fixed to 1920)
 */
void recoverVideo(unsigned char *videoR, unsigned char *videoG, unsigned char *videoB, int *vrIdx, int N, int H, int W) {
    // diffMat[i * N + j] : difference between frame i and j (i != j)
    float *diffMat = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float dImg = 0;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int dPixel = 0, d;
                    d = (int)videoR[(i * H + h) * W + w] - (int)videoR[(j * H + h) * W + w];
                    dPixel += d * d;
                    d = (int)videoG[(i * H + h) * W + w] - (int)videoG[(j * H + h) * W + w];
                    dPixel += d * d;
                    d = (int)videoB[(i * H + h) * W + w] - (int)videoB[(j * H + h) * W + w];
                    dPixel += d * d;
                    dImg += sqrt((float)dPixel);
                }
            }
            diffMat[i * N + j] = dImg;
            diffMat[j * N + i] = dImg;
        }
    }

    // used[i] : 1 if frame i is already ordered, 0 otherwise
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

    free(diffMat);
    free(used);
}
