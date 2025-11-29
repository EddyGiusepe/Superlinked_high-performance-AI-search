#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Script quickstart.py
====================
Este script demonstra como usar o Superlinked para criar um sistema
de busca semântica (similarity search) para avaliações de filmes.

O que o script faz:
------------------
1. Define um esquema (schema) para reviews de filmes
2. Cria um espaço vetorial usando embeddings de texto
3. Indexa os dados para busca rápida
4. Executa uma query de similaridade semântica
5. Retorna os resultados mais similares à busca

Conceitos principais:
--------------------
- Schema: Define a estrutura dos dados (como uma tabela de banco de dados)
- Space: Espaço vetorial onde os textos são convertidos em embeddings
- Index: Estrutura de dados otimizada para busca rápida
- Query: Define como queremos buscar e recuperar os dados

RUN
---
uv run quickstart.py
"""
from typing import List, Dict, Any
from superlinked import framework as sl

class Review(sl.Schema):
    """
    Schema que define a estrutura de uma review de filme.
    
    Attributes:
        id (sl.IdField): Identificador único da review
        text (sl.String): Texto da avaliação do filme
    """
    id: sl.IdField  
    text: sl.String


def create_search_index(model_name: str = "all-MiniLM-L6-v2") -> tuple[sl.Index, sl.Query, Review]:
    """
    Cria o índice de busca e a query configurada.
    
    Args:
        model_name: Nome do modelo de embedding a ser usado.
                   O modelo "all-MiniLM-L6-v2" é leve e eficiente para textos curtos.
                   
                   Modelos locais suportados (Sentence Transformers):
                   - all-MiniLM-L6-v2: 384 dim, rápido, bom para início
                   - all-mpnet-base-v2: 768 dim, melhor qualidade, mais lento
                   - paraphrase-multilingual-MiniLM-L12-v2: multilíngue
                   
                   Para usar OpenAI (text-embedding-3-small/large):
                   - Veja: one_quickstart/quickstart_openai.py
                   - Requer: API key e integração customizada
    
    Returns:
        tuple contendo:
        - index: Índice para armazenar e buscar vetores
        - query: Query configurada para busca por similaridade
        - review: Instância do schema Review
    """
    # Instancia o schema
    review = Review()
    
    # Cria um espaço vetorial de similaridade textual
    # Converte textos em vetores numéricos (embeddings) usando o modelo especificado
    space = sl.TextSimilaritySpace(
        text=review.text, 
        model=model_name
    )
    
    # Cria um índice para armazenar e buscar eficientemente os vetores
    index = sl.Index(space)
    
    # Define a query: busca reviews similares ao parâmetro "search"
    # .find(review): o que queremos buscar (objetos do tipo Review)
    # .similar(space, ...): busca por similaridade no espaço vetorial
    # .select_all(): retorna todos os campos do schema
    query = sl.Query(index).find(review).similar(
        space, 
        sl.Param("search")
    ).select_all()
    
    return index, query, review


def setup_executor(
    review_schema: Review, 
    index: sl.Index
) -> tuple[sl.InMemoryExecutor, sl.InMemorySource]:
    """
    Configura o executor e a fonte de dados em memória.
    
    Args:
        review_schema: Schema das reviews
        index: Índice de busca criado anteriormente
    
    Returns:
        tuple contendo:
        - app: Executor configurado e em execução
        - source: Fonte de dados onde inserimos as reviews
    """
    # Fonte de dados em memória (para produção, poderia ser um banco de dados)
    source = sl.InMemorySource(review_schema)
    
    # Executor que processa queries e mantém os índices atualizados
    app = sl.InMemoryExecutor(
        sources=[source], 
        indices=[index]
    ).run()
    
    return app, source


def add_sample_reviews(source: sl.InMemorySource) -> None:
    """
    Adiciona reviews de exemplo à fonte de dados.
    
    Args:
        source: Fonte de dados onde as reviews serão inseridas
    """
    reviews_data: List[Dict[str, str]] = [
        {
            "id": "1", 
            "text": "Amazing acting and great story"
        },
        {
            "id": "2", 
            "text": "Boring plot with bad acting"
        }
    ]
    
    # Insere os dados na fonte
    # Automaticamente gera embeddings e atualiza o índice
    source.put(reviews_data)


def search_reviews(
    app: sl.InMemoryExecutor, 
    query: sl.Query, 
    search_text: str
) -> Any:
    """
    Executa uma busca por reviews similares ao texto fornecido.
    
    Args:
        app: Executor configurado
        query: Query de busca
        search_text: Texto para buscar reviews similares
    
    Returns:
        DataFrame pandas com os resultados ordenados por similaridade
    """
    # Executa a query com o parâmetro de busca
    # O Superlinked converte o texto em embedding e busca os mais similares
    result = app.query(query, search=search_text)
    
    # Converte o resultado para DataFrame pandas para fácil visualização
    return sl.PandasConverter.to_pandas(result)


def main() -> None:
    """
    Função principal que orquestra todo o fluxo de busca semântica.
    """
    # 1. Criar índice e query
    index, query, review_schema = create_search_index()
    
    # 2. Configurar executor e fonte de dados
    app, source = setup_executor(review_schema, index)
    
    # 3. Adicionar dados de exemplo
    add_sample_reviews(source)
    
    # 4. Executar busca semântica
    # Busca reviews similares a "excellent performance"
    # Mesmo que não contenha as palavras exatas, encontra contextos similares
    search_term = "excellent performance"
    print(f"\n🔍 Buscando reviews similares a: '{search_term}'\n")
    print("=" * 60)
    
    results = search_reviews(app, query, search_term)
    print(results)
    print("\n💡 Note que 'Amazing acting' tem maior score de similaridade!")
    print("   Isso demonstra busca semântica, não apenas palavras-chave.")


if __name__ == "__main__":
    main()
