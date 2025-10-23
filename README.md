# 🛍️ Sistema de Recomendação de Produtos por Imagem (Similitude Visual)

Este projeto implementa um sistema de recomendação de produtos baseado em **similitude visual**. Ele utiliza um modelo de Visão Computacional de última geração (CLIP) para extrair características (embeddings) de imagens de produtos e, em seguida, usa o algoritmo FAISS para encontrar os itens visualmente mais semelhantes de forma rápida e eficiente.

## ✨ Como Funciona

1. **Embedding de Imagens:** O modelo **CLIP** (da OpenAI) converte cada imagem de produto em um vetor numérico de alta dimensão. Produtos visualmente semelhantes (ex: todas as camisetas azuis) terão vetores próximos no espaço vetorial.
2. **Indexação Rápida:** O **FAISS** (Facebook AI Similarity Search) é usado para indexar esses vetores, permitindo buscas de similaridade em tempo real.
3. **Recomendação:** Dada uma imagem de produto de referência, o sistema localiza os "vizinhos mais próximos" no espaço vetorial, que são os produtos mais similares para recomendação.

## ⚙️ Pré-requisitos

Para executar este projeto, você precisa ter o Python instalado e as seguintes bibliotecas:

```bash
pip install torch torchvision transformers Pillow scikit-learn numpy faiss-cpu matplotlib
```
## 🚀 Como Executar
1. Preparação dos Dados
Crie uma pasta chamada img na raiz do projeto e adicione todas as imagens dos seus produtos que você deseja indexar.
```bash
mkdir img
# Coloque aqui seus arquivos: ./img/tenis.jpg, ./img/vestido.png, etc.
```

2. Execução do Script
Execute o script principal em seu ambiente Python:
```bash
python product_recommender.py
```

3. Resultados
O script irá:

Carregar o modelo CLIP.

Calcular o vetor (embedding) para todas as imagens na pasta img.

Escolherá um produto aleatório da pasta img como item de referência.

Mostrará um gráfico com o Produto de Referência e os K produtos mais similares encontrados.

Você pode ajustar a variável K e a forma como o produto de referência é escolhido dentro do arquivo product_recommender.py.



📌 Principais Tecnologias
Python: Linguagem de programação.

CLIP (Hugging Face): Modelo para geração de embeddings de imagens.

FAISS: Biblioteca para busca eficiente de vizinhos mais próximos.

Matplotlib: Para visualização e plotagem das recomendações.

PIL (Pillow): Para manipulação de imagens.



🤝 Contribuições
Sinta-se à vontade para abrir issues ou enviar pull requests para melhorar este projeto!
