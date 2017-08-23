__kernel void recover_video(__global unsigned char* R,
                       __global unsigned char* G,
                       __global unsigned char* B,
                       int N,
                       int H,
                       int W,
                       __global float* diffMat) {

    __private int i = get_global_id(0);
    __private int j = get_global_id(1);

    __private int _w, _h;
    __private float dImg = 0;
    for (_h = 0; _h < H; _h++) {
        for (_w = 0; _w < W; _w++) {
            __private int dPixel = 0;
            __private int d = (int)R[(i * H + _h) * W + _w] - (int)R[(j * H + _h) * W + _w];
            dPixel += d * d;
            d = (int)G[(i * H + _h) * W + _w] - (int)G[(j * H + _h) * W + _w];
            dPixel += d * d;
            d = (int)B[(i * H + _h) * W + _w] - (int)B[(j * H + _h) * W + _w];
            dPixel += d * d;
            dImg += sqrt((float) dPixel);
        }
    }

    diffMat[i * N + j] = dImg;
    diffMat[j * N + i] = dImg;
}
