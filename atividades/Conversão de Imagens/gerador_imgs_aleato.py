import os
import random
from PIL import Image

# Diretório para salvar as imagens
#OUTPUT_DIR = "random_images" # Alterar para fazer o output de imagens.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data set de imagens") 

# Função para criar imagens aleatórias
def generate_random_image(image_id):
    # Gerar dimensões aleatórias entre 500x500
    width = random.randint(500, 500)
    height = random.randint(500, 500)
    
    # Criar uma nova imagem RGB
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    # Preencher cada pixel com uma cor aleatória
    for x in range(width):
        for y in range(height):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            pixels[x, y] = (r, g, b)
    
    # Salvar a imagem no diretório de saída
    output_path = os.path.join(OUTPUT_DIR, f"random_image_{image_id}.png")
    img.save(output_path)
    print(f"Imagem gerada: {output_path}")

# Função principal para gerar várias imagens
def generate_images(num_images):
    # Garantir que o diretório de saída exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Gerar as imagens
    for i in range(1, num_images + 1):
        generate_random_image(i)

# Quantidade de imagens a serem geradas
NUM_IMAGES = int(input("Digite a quantidade de imagens a serem geradas: "))

if __name__ == "__main__":
    generate_images(NUM_IMAGES)
