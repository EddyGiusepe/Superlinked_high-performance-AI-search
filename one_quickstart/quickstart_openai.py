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
import json
from superlinked import framework as sl

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


class Product(sl.Schema):
    id: sl.IdField
    description: sl.String
    rating: sl.Integer


product = Product()

description_space = sl.TextSimilaritySpace(text=product.description,
                                           model="Alibaba-NLP/gte-large-en-v1.5"
                                          )
rating_space = sl.NumberSpace(number=product.rating,
                              min_value=1,
                              max_value=5,
                              mode=sl.Mode.MAXIMUM
                             )
index = sl.Index([description_space, rating_space], fields=[product.rating], temperature=0.0)


# Define your query and parameters to set them directly at query-time
# or let an LLM fill them in for you using the `natural_language_query` param.
# Don't forget to set your OpenAI API key to unlock this feature.
query = (
    sl.Query(
        index,
        weights={
            description_space: sl.Param("description_weight"),
            rating_space: sl.Param("rating_weight"),
        },
    )
    .find(product)
    .similar(
        description_space,
        sl.Param(
            "description_query",
            description="The text in the user's query that refers to product descriptions.",
        ),
    )
    .limit(sl.Param("limit"))
    .with_natural_query(
        sl.Param("natural_language_query"),
        sl.OpenAIClientConfig(api_key=OPENAI_API_KEY, model="gpt-5-nano")
    )
)

# Run the app in-memory (server & Apache Spark executors available too!).
source = sl.InMemorySource(product)
executor = sl.InMemoryExecutor(sources=[source], indices=[index])
app = executor.run()


# Ingest data into the system - index updates and other processing happens automatically.
source.put([
    {
        "id": 1,
        "description": "Budget toothbrush in black color. Just what you need.",
        "rating": 1,
    },
    {
        "id": 2,
        "description": "High-end toothbrush created with no compromises.",
        "rating": 5,
    },
    {
        "id": 3,
        "description": "A toothbrush created for the smart 21st century man.",
        "rating": 3,
    },
])

result = app.query(query, natural_query="best toothbrushes", limit=1)

# Examine the extracted parameters from your query
print(json.dumps(result.metadata, indent=2))

# The result is the 5-star rated product.
sl.PandasConverter.to_pandas(result)