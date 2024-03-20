#include <stdio.h>
#include <cuda_runtime.h>
#define times 256
// CUDA 初始化
void cudaInitialize(int deviceId) {
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        exit(EXIT_FAILURE);
    }
}

// 计算带宽的函数
void calculateBandwidth(size_t bytes, float timeMs) {
    float bandwidth =times * (bytes / 1e6) / (timeMs / 1e3); // MB/s
    printf("Bandwidth: %f MB/s\n", bandwidth);
}

int main() {
    // CUDA 初始化
    cudaInitialize(0);

    // 数据大小
    size_t dataSize = 1536 * 1024 * 1024;
    // 分配主机内存
    char *h_data = (char *)malloc(dataSize);

    // 分配设备内存
    char *d_data;
    cudaMallocPitch((void **)&d_data, dataSize);

    // 填充主机内存
    #pragma omp parallel for
    for (size_t i = 0; i < dataSize; ++i) {
        h_data[i] = (char)i;
    }

    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 将数据从主机复制到设备
    for(int i=0;i<times;i++)
    {
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    }
    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算时间
    float elapsedTimeMs = 0.0f;
    cudaEventElapsedTime(&elapsedTimeMs, start, stop);

    // 计算带宽
    calculateBandwidth(dataSize, elapsedTimeMs);

    // 释放资源
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
