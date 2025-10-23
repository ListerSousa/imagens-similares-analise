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
