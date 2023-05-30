
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pgm.h"
#include "bmp.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
# define M_PI           3.14159265358979323846
const float radInc = degreeInc * M_PI / 180;

#define THREADS_PER_BLOCK 1024

//macro para imprimir errores
#define CUDA_ERR_MACRO(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran(unsigned char* pic, int w, int h, int** acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
    memset(*acc, 0, sizeof(int) * rBins * degreeBins); //init en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++) //por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pic[idx] > 0) //si pasa thresh, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;  // y-coord has to be reversed
                float theta = 0;         // actual angle
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                    theta += radInc;
                }
            }
        }
}
#define sharedmem 1
#define constantmem 1
//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
#if constantmem
__constant__ float d_constCos[degreeBins];
__constant__ float d_constSin[degreeBins];


//*****************************************************************
//TODO Kernel memoria compartida
__global__ void GPU_HoughTranConstShared(unsigned char* pic, int w, int h,
    int* out, float rMax, float rScale)
{
    __shared__ int acc[rBins * degreeBins];
    constexpr int offset = (rBins * degreeBins) / THREADS_PER_BLOCK;

    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h) return;      // in case of extra threads in block

    for (int i = 0; i < offset; i++)
        acc[threadIdx.x + THREADS_PER_BLOCK * i] = 0;
    if (threadIdx.x < (rBins * degreeBins) % THREADS_PER_BLOCK)
        acc[(THREADS_PER_BLOCK)*offset + threadIdx.x] = 0;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;


    __syncthreads();

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            float r = xCoord * d_constCos[tIdx] + yCoord * d_constSin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
    __syncthreads();


    for (int i = 0; i < offset; i++)
        atomicAdd(out + threadIdx.x + THREADS_PER_BLOCK * i,
            acc[threadIdx.x + THREADS_PER_BLOCK * i]);

    if (threadIdx.x < (rBins * degreeBins) % THREADS_PER_BLOCK)
        atomicAdd(out + (THREADS_PER_BLOCK)*offset + threadIdx.x,
            acc[(THREADS_PER_BLOCK)*offset + threadIdx.x]);

}


//TODO Kernel memoria Constante
__global__ void GPU_HoughTranConst(unsigned char* pic, int w, int h,
    int* acc, float rMax, float rScale)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h) return;      // in case of extra threads in block

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    //TODO eventualmente usar memoria compartida para el acumulador

    if (pic[gloID] == 0)
        return;

    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
        //TODO utilizar memoria constante para senos y cosenos
        float r = xCoord * d_constCos[tIdx] + yCoord * d_constSin[tIdx];
        int rIdx = (r + rMax) / rScale;
        //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
        //R: porque el acumulador no es un vector en el espacio de los pixels, sino en el espacio de pesos para líneas
        //Las threads no tienen una relación 1 a 1 con la memoria en este espacio.
        atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
}
#endif

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char* pic, int w, int h, 
            int* acc, float rMax, float rScale, 
            float* d_Cos, float* d_Sin)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h) return;      // in case of extra threads in block

    int xCent = w / 2;
    int yCent = h / 2;
    
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    //TODO eventualmente usar memoria compartida para el acumulador

    if (pic[gloID] == 0) 
        return;

    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
        float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
        int rIdx = (r + rMax) / rScale;
        //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
        //R: porque el acumulador no es un vector en el espacio de los pixels, sino en el espacio de pesos para líneas
        //Las threads no tienen una relación 1 a 1 con la memoria en este espacio.
        atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
}

__global__ void makeImage(unsigned char* pic, int w, int h,
    int* acc, float rMax, float rScale,
    float* d_Cos, float* d_Sin, char3* out, int thresholdMul = 6)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID > w * h) return;      // in case of extra threads in block

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    //TODO eventualmente usar memoria compartida para el acumulador

    out[gloID] = make_char3(pic[gloID], 
                            pic[gloID], 
                            pic[gloID]); //pgm y bmp guardan los pixeles de manera inversa


    __shared__ int max[THREADS_PER_BLOCK];

    max[threadIdx.x] = 0;
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
        float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
        int rIdx = (r + rMax) / rScale;

        const int val = *(acc + (rIdx * degreeBins + tIdx));
        max[threadIdx.x] = val > max[threadIdx.x] ? val : max[threadIdx.x];
    }

    if (max[threadIdx.x] > w*thresholdMul)
        out[gloID] = make_char3(   out[gloID].x   +      max[threadIdx.x] * 37 % 100,//blue
                                   out[gloID].y    +    max[threadIdx.x] * 91 % 100,//green
                                   out[gloID].z  +    max[threadIdx.x] * 51 % 100); //red
}

//*****************************************************************
int main(int argc, char** argv)
{
    int i;
    int threshold = 6;
    if (argc > 2)
        threshold = strtol(argv[2], 0, 10);

    PGMImage inImg(argv[1]);

    int* cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float* d_Cos;
    float* d_Sin;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&d_Cos, sizeof(float) * degreeBins);
    cudaMalloc((void**)&d_Sin, sizeof(float) * degreeBins);

    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);


    // pre-compute values to be stored
    float* pcCos = (float*)malloc(sizeof(float) * degreeBins);
    float* pcSin = (float*)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // TODO eventualmente volver memoria global
    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // setup and copy data from host to device
    unsigned char* d_in, * h_in; 
    char3 * d_out_pic;
    int* d_hough, * h_hough;

    h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

    h_hough = (int*)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void**)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void**)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);



    // execution configuration uses a 1-D grid of 1-D blocks, each made of THREADS_PER_BLOCK threads
    //1 thread por pixel
    int blockNum = ceil((double)w * (double)h / (double)THREADS_PER_BLOCK);

#if constantmem || sharedmem
    cudaMemcpyToSymbol(d_constCos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_constSin, pcSin, sizeof(float) * degreeBins);
#endif

    //Timed CUDA computation
    cudaEventRecord(start);
#if sharedmem
    GPU_HoughTranConstShared <<< blockNum, THREADS_PER_BLOCK >>> (d_in, w, h, d_hough, rMax, rScale);
#elif constantmem
    GPU_HoughTranConst <<< blockNum, THREADS_PER_BLOCK >>> (d_in, w, h, d_hough, rMax, rScale);
#else
    GPU_HoughTran << < blockNum, THREADS_PER_BLOCK >> > (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
#endif
    cudaEventRecord(stop);

    // get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Done in %fms!\n", milliseconds);

    cudaMalloc((void**)&d_out_pic, sizeof(unsigned char) * w * h * 3);
    cudaMemset(d_out_pic, 0, sizeof(unsigned char) * w * h * 3);
    makeImage <<< blockNum, THREADS_PER_BLOCK >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin, d_out_pic, threshold);

    unsigned char* imgBuffer = (unsigned char*)malloc(sizeof(unsigned char) * w * h * 3);
    cudaMemcpy(imgBuffer, d_out_pic, sizeof(unsigned char) * w * h * 3, cudaMemcpyDeviceToHost);

    writeBMP("foo.bmp", imgBuffer, w, h);

    // compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Checked!");

    // TODO clean-up
    cudaFree((void*)d_in);
    cudaFree((void*)d_hough);
    cudaFree((void*)d_Cos);
    cudaFree((void*)d_Sin);
    cudaFree((void*)d_out_pic);

    free(pcCos);
    free(pcSin);
    free(h_hough);

    return 0;
}
