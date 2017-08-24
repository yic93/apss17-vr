__kernel void recover_video(__global unsigned char* R,
                       __global unsigned char* G,
                       __global unsigned char* B,
                       int N,
                       int H,
                       int W,
//                       __global float* diffMat,
                       __local float* local_mem,
                       __global float* diffFrameMat,
                       int queue_number) {

    __private int2 group_id = (int2) (get_group_id(0), get_group_id(1));
    if (group_id.x >= group_id.y) return;
    if(queue_number == 0 && ((group_id.x/32 > group_id.y/16) || (group_id.x/32 + group_id.y/16 < 1))) return;
    if(queue_number == 1 && ((group_id.x/32 < group_id.y/16) || (group_id.x/32 + group_id.y/16 > 1))) return;
    __private int2 global_id = (int2) (get_global_id(0), get_global_id(1));
    __private int i = global_id.x / 1920;
    __private int j = global_id.y / 1080;
    __private int _w = global_id.x % 1920;
    __private int _h = global_id.y % 1080;
    __private int l_i = get_local_id(0);
    __private int l_j = get_local_id(1);

    __private int dPixel = 0;
    __private int d = (int)R[(i * H + _h) * W + _w] - (int)R[(j * H + _h) * W + _w];
    dPixel += d * d;
    d = (int)G[(i * H + _h) * W + _w] - (int)G[(j * H + _h) * W + _w];
    dPixel += d * d;
    d = (int)B[(i * H + _h) * W + _w] - (int)B[(j * H + _h) * W + _w];
    dPixel += d * d;

    __private float dImg = sqrt((float) dPixel);
    
    local_mem[l_j * 32 + l_i] = dImg;
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
    for (__private int p = 16; p >= 1; p = p >> 1) {
        if (l_i < p) local_mem[l_j * 32 + l_i] += local_mem[l_j * 32 + l_i + p];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    */

    // unloop
    if (l_i < 16) local_mem[l_j * 32 + l_i] += local_mem[l_j * 32 + l_i + 16];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i < 8) local_mem[l_j * 32 + l_i] += local_mem[l_j * 32 + l_i + 8];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i < 4) local_mem[l_j * 32 + l_i] += local_mem[l_j * 32 + l_i + 4];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i < 2) local_mem[l_j * 32 + l_i] += local_mem[l_j * 32 + l_i + 2];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i < 1) local_mem[l_j * 32 + l_i] += local_mem[l_j * 32 + l_i + 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
    for (__private int p = 8; p >= 1; p = p >> 1) {
        if (l_i == 0 && l_j < p) local_mem[l_j * 32] += local_mem[(l_j + p) * 32];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    */

    // unloop
    if (l_i == 0 && l_j < 9) local_mem[l_j * 32] += local_mem[(l_j + 9) * 32];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i == 0 && l_j < 4) local_mem[l_j * 32] += local_mem[(l_j + 4) * 32];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i == 0 && l_j < 2) local_mem[l_j * 32] += local_mem[(l_j + 2) * 32];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i == 0 && l_j < 1) local_mem[l_j * 32] += local_mem[(l_j + 1) * 32];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i == 0 && l_j == 0) {
        __private int2 index_of_group = group_id / 60;
        __private int2 index_in_group = group_id % 60;
        __private int frame_mat_index1 = ((index_of_group.y * 60 + index_in_group.y) * 60 * N) + (60 * index_of_group.x + index_in_group.x);
        __private int frame_mat_index2 = ((index_of_group.x * 60 + index_in_group.x) * 60 * N) + (60 * index_of_group.y + index_in_group.y);

        local_mem[0] += local_mem[8 * 32];
        diffFrameMat[frame_mat_index1] = local_mem[0];
        diffFrameMat[frame_mat_index2] = local_mem[0];
    }
    return;
}
