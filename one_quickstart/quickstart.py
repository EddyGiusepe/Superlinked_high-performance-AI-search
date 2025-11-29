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
import os
from typing import List, Dict, Any
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
        rating (sl.Float): Pontuação numérica do filme (0.0 a 5.0)
    """
    id: sl.IdField  
    text: sl.String
    rating: sl.Float


def create_search_index(
    model_name: str = "all-MiniLM-L6-v2",
    text_weight: float = 0.7,
    rating_weight: float = 0.3,
    enable_natural_query: bool = False
) -> tuple[sl.Index, sl.Query, Review]:
    """
    Cria o índice de busca e a query configurada com MÚLTIPLOS ESPAÇOS VETORIAIS.
    
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
        
        text_weight: Peso para o espaço textual (padrão: 0.7 = 70%)
        rating_weight: Peso para o espaço numérico (padrão: 0.3 = 30%)
        enable_natural_query: Se True, habilita queries em linguagem natural usando OpenAI.
                             Requer a variável de ambiente OPENAI_API_KEY.
    
    Returns:
        tuple contendo:
        - index: Índice para armazenar e buscar vetores
        - query: Query configurada para busca por similaridade
        - review: Instância do schema Review
    """
    # Instancia o schema
    review = Review()
    
    # 1. ESPAÇO TEXTUAL: Cria um espaço vetorial de similaridade textual
    # Converte textos em vetores numéricos (embeddings) usando o modelo especificado
    text_space = sl.TextSimilaritySpace(
        text=review.text, 
        model=model_name
    )
    
    # 2. ESPAÇO NUMÉRICO: Cria um espaço para similaridade de ratings
    # Filmes com ratings próximos terão maior similaridade
    rating_space = sl.NumberSpace(
        number=review.rating,
        min_value=0.0,
        max_value=5.0,
        mode=sl.Mode.SIMILAR  # Busca valores similares (próximos)
    )
    
    # 3. CRIA ÍNDICE COM MÚLTIPLOS ESPAÇOS
    # Isso é o PODER do Superlinked! 🚀
    # O Index recebe uma LISTA de espaços vetoriais
    index = sl.Index([text_space, rating_space])
    
    # 4. DEFINE A QUERY COM PESOS PARA CADA ESPAÇO
    # Os pesos controlam a importância de cada critério de busca
    # IMPORTANTE: Pesos são aplicados na Query, não no Index!
    
    if enable_natural_query:
        # MODO NATURAL QUERY: Usa LLM para extrair parâmetros automaticamente
        # Exemplo: "A film with incredible acting and a rating above 4"
        # O LLM extrai: search_text="incredible acting", search_rating=4.0
        query = (
            sl.Query(
                index,
                weights={
                    text_space: text_weight,
                    rating_space: rating_weight
                }
            )
            .find(review)
            .similar(
                text_space, 
                sl.Param(
                    "search_text",
                    description="The text describing the movie qualities (acting, story, etc.)"
                )
            )
            .similar(
                rating_space, 
                sl.Param(
                    "search_rating",
                    description="The numeric rating value (0.0 to 5.0) mentioned or implied in the query"
                )
            )
            .with_natural_query(
                sl.Param("natural_query"),
                sl.OpenAIClientConfig(
                    api_key=OPENAI_API_KEY,
                    model="gpt-4o"  # Modelo estável que suporta todas as configs
                )
            )
            .select_all()
        )
    else:
        # MODO TRADICIONAL: Parâmetros explícitos
        query = (
            sl.Query(
                index,
                weights={
                    text_space: text_weight,      # 70% peso para similaridade textual
                    rating_space: rating_weight   # 30% peso para proximidade de rating
                }
            )
            .find(review)
            .similar(text_space, sl.Param("search_text"))
            .similar(rating_space, sl.Param("search_rating"))
            .select_all()
        )
    
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
    Adiciona reviews de exemplo à fonte de dados com TEXTO + RATING numérico.
    
    Args:
        source: Fonte de dados onde as reviews serão inseridas
    """
    reviews_data: List[Dict[str, Any]] = [
        {
            "id": "1", 
            "text": "Incredible performances and a great story, masterpiece of cinema!",
            "rating": 4.5
        },
        {
            "id": "2", 
            "text": "A tedious plot with bad acting, complete waste of time.",
            "rating": 1.5
        },
        {
            "id": "3",
            "text": "Amazing visual effects but the story could be better.",
            "rating": 3.5
        },
        {
            "id": "4",
            "text": "One of the best films I've ever seen, absolutely brilliant!",
            "rating": 5.0
        },
        {
            "id": "5",
            "text": "Mediocre acting and a predictable storyline.",
            "rating": 2.0
        },
        {
            "id": "6",
            "text": "Outstanding direction and phenomenal performances throughout.",
            "rating": 4.8
        },
        {
            "id": "7",
            "text": "Boring and uninspired, couldn't even finish watching.",
            "rating": 1.0
        },
        {
            "id": "8",
            "text": "Decent movie, entertaining but nothing special.",
            "rating": 3.0
        }
    ]
    
    # Insere os dados na fonte
    # Automaticamente gera embeddings para TEXTO e normaliza RATINGS
    source.put(reviews_data)


def search_reviews(
    app: sl.InMemoryExecutor, 
    query: sl.Query, 
    search_text: str,
    search_rating: float
) -> Any:
    """
    Executa uma busca por reviews similares usando TEXTO + RATING (modo tradicional).
    
    Esta é a MAGIA do Superlinked! 🎯
    A busca combina:
    - Similaridade semântica do texto
    - Proximidade do rating numérico
    
    Args:
        app: Executor configurado
        query: Query de busca
        search_text: Texto para buscar reviews similares
        search_rating: Rating para buscar reviews com valores próximos
    
    Returns:
        DataFrame pandas com os resultados ordenados por similaridade combinada
    """
    # Executa a query com AMBOS os parâmetros
    # O Superlinked:
    # 1. Converte o texto em embedding
    # 2. Normaliza o rating numérico
    # 3. Combina ambas as similaridades com os pesos definidos
    # 4. Retorna os resultados mais similares
    result = app.query(
        query, 
        search_text=search_text,
        search_rating=search_rating
    )
    
    # Converte o resultado para DataFrame pandas para fácil visualização
    return sl.PandasConverter.to_pandas(result)


def search_reviews_natural(
    app: sl.InMemoryExecutor, 
    query: sl.Query, 
    natural_query: str
) -> Any:
    """
    Executa uma busca usando LINGUAGEM NATURAL! 🚀
    
    O LLM (GPT) extrai automaticamente:
    - O texto descritivo (ex: "incredible acting")
    - O rating numérico (ex: 4.0 de "rating above 4")
    
    Exemplos de queries naturais:
    - "A film with incredible acting and a rating above 4"
    - "I want terrible movies with ratings below 2"
    - "Show me decent films around 3 stars"
    
    Args:
        app: Executor configurado
        query: Query de busca (deve ter natural_query habilitado)
        natural_query: Query em linguagem natural
    
    Returns:
        DataFrame pandas com os resultados
    """
    result = app.query(query, natural_query=natural_query)
    
    # Mostra os parâmetros extraídos pelo LLM (sem embeddings)
    if hasattr(result, 'metadata') and result.metadata:
        print("\n🤖 Parâmetros extraídos pelo LLM:")
        try:
            metadata = result.metadata
            if hasattr(metadata, 'model_dump'):
                data = metadata.model_dump()
                # Extrai apenas search_params (parâmetros úteis)
                if 'search_params' in data:
                    params = data['search_params']
                    # Filtra apenas os parâmetros relevantes
                    print(f"   📝 Texto extraído: '{params.get('search_text', 'N/A')}'")
                    print(f"   ⭐ Rating extraído: {params.get('search_rating', 'N/A')}")
                    print(f"   💬 Query original: '{params.get('natural_query', 'N/A')}'")
        except Exception:
            pass  # Ignora erros silenciosamente
    
    return sl.PandasConverter.to_pandas(result)


def main() -> None:
    """
    Função principal que demonstra o PODER do Superlinked!
    
    Combina busca semântica (texto) + busca numérica (rating)
    para encontrar reviews que correspondem a AMBOS os critérios.
    
    Suporta dois modos:
    1. Modo Tradicional: Parâmetros explícitos (search_text + search_rating)
    2. Modo Natural Query: Query em linguagem natural (requer OPENAI_API_KEY)
    """
    # Verifica se há API key do OpenAI para habilitar natural query
    has_openai_key = bool(OPENAI_API_KEY)
    
    print("\n" + "=" * 80)
    if has_openai_key:
        print("🚀 DEMONSTRAÇÃO: Superlinked - Natural Query (Linguagem Natural)")
    else:
        print("🚀 DEMONSTRAÇÃO: Superlinked - Busca Híbrida (Texto + Numérico)")
    print("=" * 80)
    
    # 1. Criar índice e query com pesos personalizados
    print("\n📊 Criando índice com espaços combinados:")
    print("   - Espaço Textual (70%): Similaridade semântica")
    print("   - Espaço Numérico (30%): Proximidade de rating")
    
    if has_openai_key:
        print("   ✨ MODO: Natural Query habilitado (usando OpenAI GPT)")
    else:
        print("   📝 MODO: Parâmetros tradicionais")
        print("   💡 Dica: Configure OPENAI_API_KEY para usar queries naturais!")
    
    index, query, review_schema = create_search_index(
        text_weight=0.7,
        rating_weight=0.3,
        enable_natural_query=has_openai_key
    )
    
    # 2. Configurar executor e fonte de dados
    app, source = setup_executor(review_schema, index)
    
    # 3. Adicionar dados de exemplo
    print("\n📝 Adicionando 8 reviews com textos e ratings...")
    add_sample_reviews(source)
    print("   ✅ Dados indexados com sucesso!")
    
    # 4. DEMONSTRAÇÃO 1: Busca por filmes EXCELENTES
    print("\n" + "=" * 80)
    print("🎯 BUSCA 1: Filmes com reviews positivas e rating alto")
    print("=" * 80)
    
    if has_openai_key:
        # MODO NATURAL QUERY
        natural_query_1 = "A film with incredible acting and a rating above 4"
        print(f"💬 Query natural: '{natural_query_1}'")
        print("\n🔎 Resultados (ordenados por similaridade combinada):")
        print("-" * 80)
        results_1 = search_reviews_natural(app, query, natural_query_1)
    else:
        # MODO TRADICIONAL
        search_text_1 = "amazing performance great movie"
        search_rating_1 = 4.5
        print(f"📝 Texto de busca: '{search_text_1}'")
        print(f"⭐ Rating de busca: {search_rating_1}")
        print("\n🔎 Resultados (ordenados por similaridade combinada):")
        print("-" * 80)
        results_1 = search_reviews(app, query, search_text_1, search_rating_1)
    
    print(results_1.to_string(index=False))
    
    # 5. DEMONSTRAÇÃO 2: Busca por filmes RUINS
    print("\n" + "=" * 80)
    print("🎯 BUSCA 2: Filmes com reviews negativas e rating baixo")
    print("=" * 80)
    
    if has_openai_key:
        # MODO NATURAL QUERY
        natural_query_2 = "Terrible movies with bad acting and ratings below 2"
        print(f"💬 Query natural: '{natural_query_2}'")
        print("\n🔎 Resultados (ordenados por similaridade combinada):")
        print("-" * 80)
        results_2 = search_reviews_natural(app, query, natural_query_2)
    else:
        # MODO TRADICIONAL
        search_text_2 = "terrible boring bad movie"
        search_rating_2 = 1.5
        print(f"📝 Texto de busca: '{search_text_2}'")
        print(f"⭐ Rating de busca: {search_rating_2}")
        print("\n🔎 Resultados (ordenados por similaridade combinada):")
        print("-" * 80)
        results_2 = search_reviews(app, query, search_text_2, search_rating_2)
    
    print(results_2.to_string(index=False))
    
    # 6. DEMONSTRAÇÃO 3: Busca por filmes MEDIANOS
    print("\n" + "=" * 80)
    print("🎯 BUSCA 3: Filmes medianos (rating médio)")
    print("=" * 80)
    
    if has_openai_key:
        # MODO NATURAL QUERY
        natural_query_3 = "Show me decent films with average ratings around 3 stars"
        print(f"💬 Query natural: '{natural_query_3}'")
        print("\n🔎 Resultados (ordenados por similaridade combinada):")
        print("-" * 80)
        results_3 = search_reviews_natural(app, query, natural_query_3)
    else:
        # MODO TRADICIONAL
        search_text_3 = "decent average movie"
        search_rating_3 = 3.0
        print(f"📝 Texto de busca: '{search_text_3}'")
        print(f"⭐ Rating de busca: {search_rating_3}")
        print("\n🔎 Resultados (ordenados por similaridade combinada):")
        print("-" * 80)
        results_3 = search_reviews(app, query, search_text_3, search_rating_3)
    
    print(results_3.to_string(index=False))
    
    # 7. Explicação final
    print("\n" + "=" * 80)
    print("💡 O QUE VOCÊ ACABOU DE VER:")
    print("=" * 80)
    
    if has_openai_key:
        print("""
🎯 Busca com Linguagem Natural (Natural Query):
   - Escreva queries como faria ao conversar: "filmes com ótima atuação e nota acima de 4"
   - O LLM extrai automaticamente os parâmetros (texto + rating)
   - Sem necessidade de separar manualmente texto e números!
   
✨ PODER EXTRA do Modo Natural:
   - Entende contexto e intenção da query
   - Extrai valores numéricos mencionados ("acima de 4" → rating: 4.0)
   - Traduz descrições em parâmetros de busca
   - Interface mais intuitiva para usuários finais
""")
    
    print("""
🎯 Busca Híbrida Poderosa:
   - Combina SIGNIFICADO do texto (embeddings semânticos)
   - Com PROXIMIDADE numérica (ratings similares)
   
⚡ Vantagens do Superlinked:
   1. Não depende apenas de palavras-chave exatas
   2. Entende o CONTEXTO semântico das reviews
   3. Considera MÚLTIPLOS critérios simultaneamente
   4. Permite ajustar PESOS de cada espaço vetorial""")
    
    if not has_openai_key:
        print("""   5. Suporta Natural Query (configure OPENAI_API_KEY para testar!)""")
    
    print("""
🔥 Casos de Uso:
   - E-commerce: busca por produtos (descrição + preço + rating)
   - Streaming: recomendação de filmes/músicas (gênero + popularidade)
   - Imóveis: busca por casas (características + preço + localização)
   - Qualquer sistema que precisa combinar texto + números!
    """)
    
    if not has_openai_key:
        print("\n💡 QUER TESTAR NATURAL QUERY?")
        print("   Execute: export OPENAI_API_KEY='sua-chave-aqui'")
        print("   E rode o script novamente!")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
