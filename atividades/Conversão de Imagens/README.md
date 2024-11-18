# Descrição Atividade
Situação Problema elabore uma solução computacional, serial e paralela, que tem o seguinte comportamento, um data set de imagens(1, 10, 100 e 1000 imagens) será lido, a imagem deverá ter no mínimo 500x500 pixels. O programa deverá converter as imagens para preto e branco (Tons de cinza). Para implementação não será permitido o uso de funções para o processamento. De tal forma deverá realizar o processamento pixel a pixel. Para tanto deverá ser continuado, a seguinte forma/equação: r*0.298 + g*0.587 + b*0.114

## Alunos
[Arthur Rodrigues Castilho](https://github.com/ArthurRCastilho)<br>
[Cauã Cristian Inocêncio](https://github.com/CauaCristian)<br>

### Implementação dos Algoritmos

[conversao_image_serial.py](https://github.com/ArthurRCastilho/Programacao-Paralela/blob/main/atividades/Convers%C3%A3o%20de%20Imagens/serial/conversao_image_serial.py) <br>
[conversao_imagem_paralela.ipynb](https://github.com/ArthurRCastilho/Programacao-Paralela/blob/main/atividades/Convers%C3%A3o%20de%20Imagens/paralela/conversao_img_paralela.ipynb) <br>
[gerador_imgs_aleato.py](https://github.com/ArthurRCastilho/Programacao-Paralela/blob/main/atividades/Convers%C3%A3o%20de%20Imagens/gerador_imgs_aleato.py) <br>


### Modelagem em Paralela

1- Divisão por Blocos de Pixels<br>
Divida a imagem em blocos retangulares (por exemplo, cada bloco pode ter 50x50 pixels).<br>
Cada thread/processo será responsável por processar os pixels de um bloco.<br>
Calcule o valor de tons de cinza para cada pixel no bloco atribuído ao thread/processo.<br>
Após o processamento, combine os blocos para reconstruir a imagem em preto e branco.<br>
Vantagem: Simples divisão de trabalho; fácil balanceamento de carga com blocos de tamanhos iguais.<br>
Desvantagem: Pode gerar overhead de comunicação ao combinar blocos em imagens grandes.<br>

### Descrição
Levando em consideração as imagens 500x500(todas)<br>

A atividade feita em serial gastou um tempo de 127.75 segundos<br>
Já a atividade feita em paralelo gastou um total de 17.14 segundos<br>