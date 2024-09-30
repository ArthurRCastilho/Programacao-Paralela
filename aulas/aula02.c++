#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// ------------------ Aula 02 ------------------


// ------------------ Macros ------------------

// Macro para malloc (Alocação de CPU)
#define CPU_ALLOCATE(type, count) \
 (type*)malloc(sizeof(type) * (count))

// Macro para cudaMalloc (Alocação de GPU)
#define GPU_ALLOCATE(type, pointer, count) /
 cudaMalloc((void**)&pointer, sizeof(type) * (count))

// --------------------------------------------

// --------------- Funções ----------------

// Variáveis globais
int *a, *b, *c; 

// => Kernel é uma função que é executada na GPU
// kernal
__global__ void vecAdd(int* a, int* b, int* c){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
int main()
{
    // resetar recursos cuda
    cudaDeviceReset();
    // d => device == d_a => device a

    // Variáveis do lado da GPU
    int* d_a; int* d_b; int* d_c;

    int n = 256;
    int size = n * sizeof(int); // Tamanho do vetor para locação de memória

    // Alocação de memória na GPU
    // malloc do abc memoria lado CPU
    a = CPU_ALLOCATE(int, size);
    b = CPU_ALLOCATE(int, size);
    c = CPU_ALLOCATE(int, size);


    //malloc da GPU
    GPU_ALLOCATE(int, d_a, n);
    GPU_ALLOCATE(int, d_b, n);
    GPU_ALLOCATE(int, d_c, n);

    // inicializar vetor CPU
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    // mover dados da CPU para a GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // (Para onde vai, de onde vem, tamanho, tipo de movimentação)


    // Execução do Kernal
    vecAdd<<<1, n>>>(d_a, d_b, d_c);

    cubaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("\n Resultado Soma: \n");
    for (int i = 0; i < n; i++){
        printf("%d \n", c[i]);
    }

    // Liberar memória
    cudaFree(d_a);

}