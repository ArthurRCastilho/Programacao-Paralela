import os
import struct
from pathlib import Path
from PIL import Image  # Apenas para carregar e salvar imagens, processamento será manual.
import time

# Diretórios de entrada e saída
INPUT_DIR = "/Users/arthurlavidali/6 Periodo/Programação Paralela/atividades/Conversão de Imagens/data set de imagens" # Alterar para fazer o input de imagens.
OUTPUT_DIR = "/Users/arthurlavidali/6 Periodo/Programação Paralela/atividades/Conversão de Imagens/serial/imgs_convertidas"

# Função para verificar o tamanho da imagem
def is_valid_size(width, height, min_size=500):
    return width >= min_size and height >= min_size

# Função para converter a imagem para tons de cinza
def convert_to_grayscale(image_path, output_path):
    # Abrindo a imagem (modo RGB)
    with Image.open(image_path) as img:
        width, height = img.size
        pixels = img.load()
        
        # Criando uma nova imagem em tons de cinza
        gray_image = Image.new("RGB", (width, height))
        gray_pixels = gray_image.load()
        
        for x in range(width):
            for y in range(height):
                # Obter valores RGB
                r, g, b = pixels[x, y]
                # Calcular o valor do cinza
                gray = int(r * 0.298 + g * 0.587 + b * 0.114)
                # Definir o novo valor (tons de cinza RGB tem todos os valores iguais)
                gray_pixels[x, y] = (gray, gray, gray)
        
        # Salvar a nova imagem
        gray_image.save(output_path)
        print(f"Imagem convertida e salva em: {output_path}")

def process_images():
    start_time = time.time()

    # Garantir que o diretório de saída exista
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Iterar sobre as imagens do diretório de entrada
    for image_name in os.listdir(INPUT_DIR):
        image_path = os.path.join(INPUT_DIR, image_name)
        output_path = os.path.join(OUTPUT_DIR, f"gray_{image_name}")
        
        # Verificar se é uma imagem válida e processar
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if is_valid_size(width, height):
                    convert_to_grayscale(image_path, output_path)
                else:
                    print(f"Imagem {image_name} ignorada: tamanho inferior a 500x500 pixels.")
        except Exception as e:
            print(f"Erro ao processar a imagem {image_name}: {e}")
        
    end_time = time.time()
    print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    process_images()
