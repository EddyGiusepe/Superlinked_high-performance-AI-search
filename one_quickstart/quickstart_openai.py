#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script quickstart_openai.py
===========================
Este script demonstra como usar o Superlinked com embeddings da OpenAI
para busca semântica avançada de reviews de filmes.
Mas ainda aqui não vamos a usar o poder do Superlinked.

RUN
---
uv run quickstart_openai.py
"""
import os
import numpy as np
from typing import List
from openai import OpenAI
from superlinked import framework as sl

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

class Review(sl.Schema):
    """
    Schema que define a estrutura de uma review de filme.
    
    Attributes:
        id (sl.IdField): Identificador único da review
        text (sl.String): Texto da avaliação do filme
    """
    id: sl.IdField  
    text: sl.String


class OpenAIEmbeddingProvider:
    """
    Provider customizado para gerar embeddings usando a API da OpenAI.
    
    Esta classe serve como wrapper para a API de Embeddings da OpenAI,
    permitindo integração com o Superlinked.
    """
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small",
        api_key: str | None = None
    ):
        """
        Inicializa o provider de embeddings da OpenAI.
        
        Args:
            model: Nome do modelo de embedding da OpenAI
            api_key: Chave de API (se None, busca da variável de ambiente)
        
        Raises:
            ValueError: Se a chave de API não for fornecida
        """
        self.model = model
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        
        if not self.client.api_key:
            raise ValueError(
                "Chave de API da OpenAI não encontrada!\n"
                "Configure na variável de ambiente OPENAI_API_KEY"
            )
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Gera embedding para um único texto.
        
        Args:
            text: Texto para gerar embedding
        
        Returns:
            Lista de floats representando o vetor de embedding
        """
        # Remove quebras de linha (recomendação da OpenAI)
        text = text.replace("\n", " ")
        
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        
        return response.data[0].embedding
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para múltiplos textos em batch.
        
        Args:
            texts: Lista de textos para gerar embeddings
        
        Returns:
            Lista de vetores de embedding
        """
        # Remove quebras de linha
        texts = [text.replace("\n", " ") for text in texts]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        return [item.embedding for item in response.data]
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna a dimensão dos embeddings do modelo.
        
        Returns:
            Número de dimensões do vetor de embedding
        """
        # Dimensões dos modelos OpenAI
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)


def create_search_index_with_openai(
    model_name: str = "text-embedding-3-small"
) -> tuple[sl.Index, sl.Query, Review, OpenAIEmbeddingProvider]:
    """
    Cria o índice de busca usando embeddings da OpenAI.
    
    NOTA: Esta é uma implementação conceitual. O Superlinked nativamente
    suporta modelos locais via Sentence Transformers. Para usar OpenAI,
    você precisaria:
    
    1. Pré-gerar embeddings usando OpenAI API
    2. Armazenar em um vetor database (como Qdrant, Weaviate, etc.)
    3. Ou usar NumberSpace com vetores pré-computados
    
    Args:
        model_name: Nome do modelo de embedding da OpenAI
    
    Returns:
        tuple contendo index, query, review schema e embedding provider
    """
    # Inicializa o provider de embeddings da OpenAI
    embedding_provider = OpenAIEmbeddingProvider(model=model_name)
    
    # Instancia o schema:
    review = Review()
    
    # IMPORTANTE: TextSimilaritySpace do Superlinked usa modelos
    # locais (Sentence Transformers). Para usar OpenAI, precisamos de
    # uma abordagem diferente. Aqui vamos demonstrar o conceito:
    
    # Opção 1: Usar modelo local (recomendado para produção com Superlinked)
    space = sl.TextSimilaritySpace(
        text=review.text, 
        model="all-MiniLM-L6-v2"  # Modelo local
    )
    
    # TODO: Para usar OpenAI embeddings nativamente, seria necessário:
    # - Usar sl.NumberSpace com vetores pré-computados
    # - Ou integrar com vector database que suporte OpenAI embeddings
    
    index = sl.Index(space)
    
    query = sl.Query(index).find(review).similar(
        space, 
        sl.Param("search")
    ).select_all()
    
    return index, query, review, embedding_provider


def demonstrate_openai_embeddings() -> None:
    """
    Demonstra o uso direto da API de Embeddings da OpenAI.
    
    Esta função mostra como gerar embeddings usando OpenAI,
    que podem ser usados para busca semântica customizada.
    """
    print("\n" + "=" * 50)
    print("🤖 DEMONSTRAÇÃO: OpenAI Embeddings API")
    print("=" * 50 + "\n")
    
    try:
        # Inicializa o provider
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        
        # Textos de exemplo
        reviews = [
            "Atuação incrível e ótima história",
            "História chata com atuação ruim",
            "A casa de Jardim da Penha tem 3 quartos e 2 banheiros e custa 550000 reais",
            "A casa de Mata da Praia tem 4 quartos e 3 banheiros e custa 650000 reais",
            "A casa de Bairro Vermelho tem 3 quartos e 2 banheiros e custa 750000 reais",
        ]
        
        print("Reviews de exemplo:")
        for i, review in enumerate(reviews, 1):
            print(f"   {i}. {review}")
        
        print(f"\nGerando embeddings usando o modelo: {provider.model}")
        
        # Gera embeddings
        embeddings_reviews = provider.get_embeddings(reviews)
        
        print("Embeddings gerados com sucesso!")
        print(f"   - Dimensões: {len(embeddings_reviews[0])}")
        print(f"   - Primeiros 5 valores do embedding 1: {embeddings_reviews[0][:5]}")
        
        # Busca semântica
        query_text = "Procurando uma casa com um preço entre 500000 e 600000 reais"
        print(f"\nBuscando similaridade com: '{query_text}'")
        
        query_embedding = provider.get_embedding(query_text)
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calcula similaridade de cosseno entre dois vetores."""
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            return np.dot(a, b) / (a_norm * b_norm)
        
        similarities = [
            cosine_similarity(query_embedding, emb) 
            for emb in embeddings_reviews
        ]
        
        print("\nResultados de similaridade:")
        print("-" * 70)
        for review, similarity in zip(reviews, similarities):
            print(f"   Similaridade: {similarity:.6f} | Review: {review}")
        
        print("\nObservação:")
        best_match_idx = similarities.index(max(similarities))
        print(f"   Melhor match: '{reviews[best_match_idx]}'")
        print(f"   Score: {max(similarities):.6f}")
        
    except ValueError as e:
        print(f"\nErro: {e}")
        print("\nPassos para configurar:")
        print("1. Obtenha uma chave em: https://platform.openai.com/api-keys")
        print("2. Configure na variável de ambiente OPENAI_API_KEY")
        print("3. Execute novamente este script")


def main() -> None:
    """
    Função principal que demonstra o uso de OpenAI embeddings.
    """
    print("   SUPERLINKED + OPENAI EMBEDDINGS")
    
    # Demonstra uso direto da OpenAI API
    demonstrate_openai_embeddings()


if __name__ == "__main__":
    main()

