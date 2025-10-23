import os
import random
import numpy as np
import faiss
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import warnings

# Suprime avisos que o Hugging Face costuma gerar, limpando a saída
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURAÇÃO ---
IMG_FOLDER = './img'
MODEL_NAME = "openai/clip-vit-base-patch32"
K_RECOMMENDATIONS = 4 # Número de recomendações a serem exibidas (além da referência)

# --- 1. PREPARAÇÃO DO MODELO E DADOS ---

print(f"--- 1. Carregando Modelo e Preparando Dados ---")

# Carrega o modelo CLIP e o pré-processador
try:
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"ERRO ao carregar o modelo CLIP. Verifique sua conexão ou nome do modelo: {e}")
    exit()

# Lista e filtra arquivos de imagem
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
image_files = sorted([
    os.path.join(IMG_FOLDER, f) 
    for f in os.listdir(IMG_FOLDER) 
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
])

if not image_files or len(image_files) < K_RECOMMENDATIONS + 1:
    print(f"\nERRO: É necessário ter pelo menos {K_RECOMMENDATIONS + 1} imagens na pasta '{IMG_FOLDER}' para rodar a demonstração.")
    exit() 

print(f"{len(image_files)} imagens encontradas. Calculando embeddings...")


# --- 2. FUNÇÃO DE CRIAÇÃO DE EMBEDDINGS ---

def create_embeddings(image_paths, model, processor):
    """Gera o embedding (vetor numérico) para cada imagem."""
    
    embeddings = []
    product_names = [] 

    for img_path in image_paths:
        try:
            # Carrega a imagem e pré-processa
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            
            # Gera o embedding (vetor)
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=inputs.pixel_values)
            
            # Normalização (crucial para busca de similaridade)
            normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(normalized_features.squeeze().cpu().numpy())
            product_names.append(os.path.basename(img_path))
            
        except Exception as e:
            # Ignora arquivos corrompidos e continua
            print(f"AVISO: Imagem corrompida ou erro em {img_path}. Ignorando. Detalhe: {e}")
            
    # Converte para float32, necessário para o FAISS
    return np.array(embeddings).astype('float32'), product_names

# Executa a geração de embeddings
product_embeddings, product_names = create_embeddings(image_files, model, processor)

if product_embeddings.size == 0:
    print("ERRO: Nenhum embedding gerado com sucesso. Verifique os arquivos de imagem.")
    exit()

print(f"Geração de embeddings concluída. Total de vetores: {product_embeddings.shape[0]}")
D = product_embeddings.shape[1] # Dimensão do vetor


# --- 3. INDEXAÇÃO COM FAISS ---

print(f"\n--- 3. Indexando Embeddings com FAISS ---")
try:
    index = faiss.IndexFlatL2(D) 
    index.add(product_embeddings) 
    print(f"Índice FAISS criado com {index.ntotal} vetores.")
except Exception as e:
    print(f"ERRO ao criar o índice FAISS: {e}")
    exit()


# --- 4. FUNÇÃO DE RECOMENDAÇÃO E PLOTAGEM ---

def recommend_products_and_plot(reference_image_path, k, index, product_names, embed_dim):
    """Recomenda e plota os k produtos mais similares."""
    
    print(f"\n--- 4. Buscando Recomendações (K={k}) ---")
    
    # Gera o embedding da imagem de referência
    try:
        ref_img = Image.open(reference_image_path).convert("RGB")
        inputs = processor(images=ref_img, return_tensors="pt")
        with torch.no_grad():
            ref_features = model.get_image_features(pixel_values=inputs.pixel_values)
        
        # Normalização e formato FAISS
        ref_vector = (ref_features / ref_features.norm(p=2, dim=-1, keepdim=True)).squeeze().cpu().numpy().astype('float32')
            
    except Exception as e:
        print(f"ERRO ao processar imagem de referência: {e}")
        return

    # Busca os vizinhos mais próximos (k + 1 para incluir a referência)
    Dists, Indices = index.search(ref_vector.reshape(1, embed_dim), k + 1) 
    
    # Mapeia para caminhos e nomes
    all_indices = Indices.squeeze()
    image_paths_to_plot = [os.path.join(IMG_FOLDER, product_names[i]) for i in all_indices]
    
    
    # --- PLOTAGEM ---
    num_plots = k + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    
    fig.suptitle(f"Recomendação de Produtos por Similaridade Visual", fontsize=16)

    for i in range(num_plots):
        img_path = image_paths_to_plot[i]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = np.zeros((200, 200, 3), dtype=np.uint8) # Imagem placeholder em caso de erro

        axes[i].imshow(img)
        axes[i].axis('off')

        if i == 0:
            axes[i].set_title(f"PRODUTO DE REFERÊNCIA", fontsize=12, color='blue')
        else:
            rank = i 
            distance = Dists.squeeze()[i] 
            axes[i].set_title(f"Rank {rank}\nDistância: {distance:.4f}", fontsize=12, color='green')

        axes[i].text(0.5, -0.15, os.path.basename(img_path), size=10, ha="center", transform=axes[i].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.9]) 
    plt.show()


# --- 5. EXECUÇÃO PRINCIPAL ---

if __name__ == '__main__':
    # 5.1 Escolhe um produto aleatório como referência para a demonstração
    reference_product_path = random.choice(image_files)
    
    print(f"\nPRODUTO ESCOLHIDO PARA REFERÊNCIA: {os.path.basename(reference_product_path)}")
    
    # 5.2 Executa a recomendação e plota o resultado
    recommend_products_and_plot(
        reference_product_path, 
        k=K_RECOMMENDATIONS, 
        index=index, 
        product_names=product_names, 
        embed_dim=D
    )
