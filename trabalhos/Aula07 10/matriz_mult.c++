#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Função para gerar uma matriz com valores aleatórios
vector<vector<int>> gerarMatriz(int tamanho) {
    vector<vector<int>> matriz(tamanho, vector<int>(tamanho));
    for (int i = 0; i < tamanho; i++) {
        for (int j = 0; j < tamanho; j++) {
            matriz[i][j] = rand() % 100; // Gera números entre 0 e 99
        }
    }
    return matriz;
}

// Função para multiplicar duas matrizes (sem paralelismo)
vector<vector<int>> multiplicarMatrizes(const vector<vector<int>>& A, const vector<vector<int>>& B, int tamanho) {
    vector<vector<int>> C(tamanho, vector<int>(tamanho, 0));
    for (int i = 0; i < tamanho; i++) {
        for (int j = 0; j < tamanho; j++) {
            for (int k = 0; k < tamanho; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    srand(time(0)); // Para garantir que os números aleatórios sejam diferentes em cada execução
    
    // Solicita o tamanho da matriz ao usuário
    int tamanho;
    cout << "Informe o tamanho da matriz: ";
    cin >> tamanho;

    // Gerar duas matrizes com valores aleatórios
    vector<vector<int>> A = gerarMatriz(tamanho);
    vector<vector<int>> B = gerarMatriz(tamanho);

    // Medir o tempo de execução da multiplicação de matrizes (método serial)
    auto inicio = high_resolution_clock::now();
    vector<vector<int>> C = multiplicarMatrizes(A, B, tamanho);
    auto fim = high_resolution_clock::now();

    // Calcular o tempo gasto
    auto duracao = duration_cast<milliseconds>(fim - inicio).count();
    cout << "Tempo de execução (serial): " << duracao << " ms" << endl;

    return 0;
}
