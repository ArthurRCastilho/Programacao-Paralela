#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Função para gerar uma matriz com valores aleatórios
void gerarMatriz(int* matriz, int tamanho) {
    for (int i = 0; i < tamanho * tamanho; i++) {
        matriz[i] = rand() % 100; // Gera números entre 0 e 99
    }
}

// Função CUDA para multiplicar matrizes
__global__ void multiplicarMatrizesCUDA(int* A, int* B, int* C, int tamanho) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < tamanho && col < tamanho) {
        int sum = 0;
        for (int k = 0; k < tamanho; k++) {
            sum += A[row * tamanho + k] * B[k * tamanho + col];
        }
        C[row * tamanho + col] = sum;
    }
}

int main() {
    srand(time(0));

    // Solicita o tamanho da matriz ao usuário
    int tamanho;
    cout << "Informe o tamanho da matriz: ";
    cin >> tamanho;

    // Alocar memória para as matrizes no host
    int* h_A = (int*)malloc(tamanho * tamanho * sizeof(int));
    int* h_B = (int*)malloc(tamanho * tamanho * sizeof(int));
    int* h_C = (int*)malloc(tamanho * tamanho * sizeof(int));

    // Gerar matrizes aleatórias
    gerarMatriz(h_A, tamanho);
    gerarMatriz(h_B, tamanho);

    // Alocar memória para as matrizes no device (GPU)
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, tamanho * tamanho * sizeof(int));
    cudaMalloc((void**)&d_B, tamanho * tamanho * sizeof(int));
    cudaMalloc((void**)&d_C, tamanho * tamanho * sizeof(int));

    // Copiar as matrizes do host para o device (CPU -> GPU)
    cudaMemcpy(d_A, h_A, tamanho * tamanho * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, tamanho * tamanho * sizeof(int), cudaMemcpyHostToDevice);

    // Definir o número de threads por bloco e o número de blocos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((tamanho + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (tamanho + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Medir o tempo de execução da multiplicação de matrizes (paralelo)
    auto inicio = high_resolution_clock::now();

    // Chamar a função de multiplicação de matrizes na GPU
    multiplicarMatrizesCUDA<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, tamanho);

    // Esperar que todos os threads terminem
    cudaDeviceSynchronize();

    auto fim = high_resolution_clock::now();

    // Copiar o resultado de volta para o host (GPU -> CPU)
    cudaMemcpy(h_C, d_C, tamanho * tamanho * sizeof(int), cudaMemcpyDeviceToHost);

    // Calcular o tempo gasto
    auto duracao = duration_cast<milliseconds>(fim - inicio).count();
    cout << "Tempo de execução (paralelo - CUDA): " << duracao << " ms" << endl;

    // Liberar memória
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
