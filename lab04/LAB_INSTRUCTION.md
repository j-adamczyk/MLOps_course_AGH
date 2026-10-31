# Laboratory 4 instruction

## Introduction

This lab will cover vector search, vector databases, and their applications like similarity search
and Retrieval-Augmented Generation (RAG). We will work with two major approaches to implementing
this in MLOps practice: vector database (Milvus) and vector index (Postgres extension).

## PostgreSQL + TimescaleDB + pgvectorscale (2 points)

We do this in two passes: first manually in a throwaway container to see how extensions
work, then with Docker Compose to automate the same setup for real use.

### Manual setup: exploring the image

TimescaleDB is a Postgres extension for analytics and time series, but it also includes
many other useful things. TimescaleDB, in turn, can be extended with
[pgvectorscale extension](https://github.com/timescale/pgvectorscale), offering very
scalable vector search with DiskANN algorithm. And `pgvectorscale` is in turn based on
[pgvector extension](https://github.com/pgvector/pgvector), which provides vector search
to Postgres, based on HNSW algorithm.

1. Download (pull) the Docker image with TimescaleDB.
```bash
docker pull timescale/timescaledb-ha:pg16
```

2. Run the container. Note the options:
   - `-d` / `--detached` runs it in the background
   - `-p`/ `--port` maps the port 5555 of the host machine to the port 5432 of the container
   - `-e` / `--env` sets the provided environment variable name and value
```bash
docker run --rm -d --name timescaledb -p 5555:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb-ha:pg16
```

3. Connect to the PostgreSQL server using the `psql` client:
```bash
psql -d "postgres://postgres:password@localhost:5555/postgres"
```

4. At this moment, we have `pgvectorscale` extension installed on our image, as it comes preinstalled in the
   `timescaledb-ha` Docker image. Now we need to make sure can actually use it. List all enabled extensions.
   You can list the `pg_extension` table, or use the `\dx` command.
```sql
SELECT * FROM pg_extension;
```
OR
```sql
\dx
```

5. As you can see, `pgvectorscale` is not enabled, only `timescaledb`. To enable it, run:
```sql
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
```

6. Check the installed extensions again, `vectorscale` should now be listed there.

7. Stop container before moving to the next steps
```bash
docker stop timescaledb
```

### Automated setup with Docker Compose

From here on we build the real setup — reusable, declarative, ready for other services to depend on.

1. Create `vectorscale_db` directory and go there. We will prepare Docker Compose there.
2. Postgres Docker containers can run SQL script upon initialization, which is very useful
   for installing and enabling extensions. Create `initdb` directory, and
   `prepare_similarity_search_service_db.sql` file inside. Fill its contents:
```sql
-- prepare_similarity_search_service_db.sql

-- Create the database for the similarity search service
CREATE DATABASE similarity_search_service_db;

-- Connect to the database
-- TODO: switch the psql session into similarity_search_service_db
--       (hint: use the `\c <db_name>` meta-command — same family as `\dx` from step 4)

-- Enable vectorscale extension
-- TODO: enable the `vectorscale` extension inside the new database
--       (hint: same `CREATE EXTENSION IF NOT EXISTS <name> CASCADE;` you ran in step 5)
```
3. Create `docker-compose.yml` file with the contents below. It defines the `timescaledb`
   service and mounts the `initdb` script into `/docker-entrypoint-initdb.d/` inside the
   container — Postgres runs every `.sql` file in that directory on the **first** start
   (when the data volume is empty), so our database + extension are created automatically.
```yaml
services:
  timescaledb:
    image: timescale/timescaledb-ha:pg16
    container_name: vectorscaledb
    environment:
      POSTGRES_PASSWORD: password
    ports:
      - "5555:5432"
    volumes:
      - ./initdb/prepare_similarity_search_service_db.sql:/docker-entrypoint-initdb.d/prepare_similarity_search_service_db.sql
```

4. Run the container:
```bash
docker compose up -d
```

5. Your vector database is ready to use. You can execute the `psql` command and check that
   the `vectorscale` extension is enabled.

## SQLAlchemy and vector search (2 points)

### SQLAlchemy setup

For defining the table, we will use Python and [SQLAlchemy framework](https://www.sqlalchemy.org/).

A good practice is to build the database URL using the SQLAlchemy library. This option is much more
readable, safer, and easier to maintain than using the connection string directly.
```python
from sqlalchemy.engine import URL

db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="similarity_search_service_db"
)
```

Tables in SQLAlchemy are defined using class-based design. Typically, application defines
a single `Base` class, from which concrete tables inherit. They are defined quite similarly
to Pydantic, i.e. with attributes and types. However, here we also need to assign concrete
database types. To integrate it with vector search, we will also use `pgvector` library.

```python
from typing import List
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Create the base class for the table definition
class Base(DeclarativeBase):
    __abstract__ = True


# Create the table definition
class Images(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512
    
    # primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # image path - we will use it to store the path to the image file, after similarity search we can use it to retrieve the image and display it
    image_path: Mapped[str] = mapped_column(String(256))
    # image embedding - we will store the image embedding in this column, the image embedding is a list of 512 floats this is the output of the sentence transformer model
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))
```

To actually connect to the database, interact with it and run queries, we use `engine` object.
It is created with `create_engine()` function.

```python
from sqlalchemy import create_engine

engine = create_engine(db_url)
```

Now we can create the table in the database.
```python
Base.metadata.create_all(engine)
```

### SQLAlchemy queries

For testing, we need to insert some data into the table. Code below has one thing missing - 
you need to create the `Images` object based on provided data.

```python
from sqlalchemy.orm import Session
from sqlalchemy import Engine, select
import numpy as np

# reusable function to insert data into the table
def insert_image(engine: Engine, image_path: str, image_embedding: list[float]):
    with Session(engine) as session:
        # create the image object
        image = 
        # add the image object to the session
        session.add(image)
        # commit the transaction
        session.commit()

# insert some data into the table
N = 100
for i in range(N):
    image_path = f"image_{i}.jpg"
    image_embedding = np.random.rand(512).tolist()
    insert_image(engine, image_path, image_embedding)

# select first image from the table
with Session(engine) as session:
    image = session.query(Images).first()


# Note: pgvector exposes *distances*, not similarities (lower = more similar).
# To rank, sort ascending by `cosine_distance`. To filter by a similarity threshold,
# use `1 - cosine_distance > threshold`.

# order the K images by ascending cosine distance to the first image (smallest distance = most similar)
def find_k_images(engine: Engine, k: int, original_image: Images) -> list[Images]:
    with Session(engine) as session:
        # execution_options={"prebuffer_rows": True} is used to prebuffer the rows, this is useful when we want to fetch the rows in chunks and return them after session is closed
        result = session.execute(
            select(Images)
            .order_by(Images.image_embedding.cosine_distance(original_image.image_embedding))
            .limit(k),
            execution_options={"prebuffer_rows": True}
        ).scalars().all()
        return result

# find the 10 most similar images to the first image
k = 10
similar_images = find_k_images(engine, k, image)
```

### Filtering

We can filter the found nearest neighbors, e.g. by required minimal similarity score.

```python
# find images with similarity to the original above the threshold (similarity = 1 - cosine_distance)
def find_images_with_similarity_score_greater_than(engine: Engine, similarity_score: float, original_image: Images) -> list[Images]:
    with Session(engine) as session:
        result = session.execute(
            select(Images)
            .filter((1 - Images.image_embedding.cosine_distance(original_image.image_embedding)) > similarity_score),
            execution_options={"prebuffer_rows": True}
        ).scalars().all()
        return result
```

For more filtering, we will need an actual dataset with rich metadata to filter by. For this, we will utilize
[Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset). It is hosted on
HuggingFace Hub, and we can download it with:
```python
from datasets import load_dataset

dataset = load_dataset("FronkonGames/steam-games-dataset")

# get columns names and types
columns = dataset["train"].features
print(columns)

columns_to_keep = ["name", "windows", "linux", "mac", "detailed_description", "supported_languages", "price"]

N = 40000 # you can adjust this number
dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))
```

We will use columns:
* `name`
* `detailed_description`
* `price`
* `platforms` - the platforms on which the game is available; note that there is separate field
  for each platform (Windows, Linux, macOS)

For vector search, we can use the `detailed_description` column, which is an arbitrary text description. A great model
for this purpose is `distiluse-base-multilingual-cased-v2` from [Sentence Transformers](https://sbert.net/).
This model is a multilingual text transformers, and thus it will work well for descriptions of games in languages
other than English.

The `distiluse-base-multilingual-cased-v2` model supports over 50 languages. This version is a distilled
multilingual knowledge model derived from the original Universal Sentence Encoder, which only supported
15 languages. While the v2 model supports a wider range of languages, it is noted that its performance may
be a bit lower, compared to the original model for the languages it shares with v1, particularly the first
15 languages that were supported originally. Read the paper if you're interested: https://arxiv.org/pdf/2004.09813.

When looking at https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2, one can see
that it produces 512-dimensional embeddings. This is a value we need to declare in the table definition
as vector length.

```python
from sqlalchemy import Integer, Float, Boolean, Text


class Games(Base):
    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}
    
    # the vector size produced by the model taken from documentation https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    VECTOR_LENGTH = # check the model output dimensionality 
        
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    # `Text` is an unbounded string in Postgres — game descriptions can run a few KB
    description: Mapped[str] = mapped_column(Text)
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: # fill it in proper way

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
```

Let's prepare the function that will generate the embeddings for the games descriptions.
```python
from sentence_transformers import SentenceTransformer


checkpoint = "distiluse-base-multilingual-cased-v2"
model = SentenceTransformer(checkpoint)


def generate_embeddings(text: str) -> list[float]:
    return model.encode(text)
```

Now let's prepare the function that will insert the data into the table.
```python
from tqdm import tqdm


def insert_games(engine, dataset):
    with tqdm(total=len(dataset)) as pbar:
       for i, game in enumerate(dataset):
           game_description = game["detailed_description"] or ""
           game_embedding = generate_embeddings(game_description)
           name, windows, linux, mac, price = game["name"], game["windows"], game["linux"], game["mac"], game["price"]
           # keep rows with the required fields; windows/linux/mac are platform flags (True/False),
           # not missing-data signals — don't gate the insert on them or you discard most of the dataset
           if name and game_description and price is not None:
               game = Games(
                   name=game["name"], 
                   description=game_description,
                   windows=game["windows"], 
                   linux=game["linux"], 
                   mac=game["mac"], 
                   price=game["price"], 
                   game_description_embedding=game_embedding
               )
               with Session(engine) as session:
                   session.add(game)
                   session.commit()
           pbar.update(1)
```

Now we can insert the data into the table.
```python
insert_games(engine, dataset)
```

Now the function that will find the games similar to the given game, and also include given filtering criteria.
Note that pgvector exposes *distances* (lower = more similar), so ascending `ORDER BY cosine_distance(...)`
returns the closest matches first — no need for `DESC`.
```python
def find_game(
    engine: Engine, 
    game_description: str, 
    windows: Optional[bool] = None, 
    linux: Optional[bool] = None,
    mac: Optional[bool] = None,
    price: Optional[int] = None
):
    with Session(engine) as session:
        game_embedding = # generate game embedding
    
        query = (
            select(Games)
            .order_by(Games.game_description_embedding.cosine_distance(game_embedding))
        )
        
        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)
        
        result = session.execute(query, execution_options={"prebuffer_rows": True})
        game = result.scalars().first()
        
        return game
```

Our first vector search service is ready to use! Let's check it out.
```python
game = find_game(engine, "This is a game about a hero who saves the world", price=10)
print(f"Game: {game.name}")
print(f"Description: {game.description}")

game = find_game(engine, game_description="Home decorating", price=20)
print(f"Game: {game.name}")
print(f"Description: {game.description}")
```

Let's change the filtering requirements:
```python
game = find_game(engine, game_description="Home decorating", mac=True, price=5)
print(f"Game: {game.name}")
print(f"Description: {game.description}")
```

As you can see, while changing criteria to more strict, results can vary. This is the consequence of
a few things: how attribute filtering reduces the results, how model interprets the similarity of
descriptions, and how those two things interact together. A major advantage of this approach is its
overall simplicity.

## Retrieval-Augmented Generation (RAG) service (1 point)

### Vector database setup

In this part of laboratory, we will build a RAG service. It enhances the LLM text generation
capabilities with context and information drawn from a knowledge base. Relevant textual information
is found with vector search and appended to the prompt, resulting in less hallucinations and
more precise, relevant answers.

In such cases, we don't really need any additional capabilities like attributes filtering, ACID,
JOINs or other Postgres-related advantages. Thus, we will use [Milvus](https://milvus.io/), a typical
example of vector database. To generate embeddings, we will use
[Silver Retriever model](https://huggingface.co/ipipan/silver-retriever-base-v1.1) from Sentence Transformers.
It is based on HerBERT model for Polish language, and finetuned for retrieval of similar vectors.

1. Start by installing [Milvus Lite](https://milvus.io/docs/milvus_lite.md). It runs inside your Python
   process and stores data in a local file:
```bash
mkdir milvus_db
cd milvus_db

uv add "pymilvus[milvus-lite]"
```
2. Next code sections are quite interactive and will probably be easier to run inside a Jupyter
   Notebook. Start it with `jupyter notebook`.
3. Let's connect to the database. Passing a local file path to `MilvusClient` creates the database
   in that file.
```python
from pymilvus import MilvusClient

milvus_client = MilvusClient("./milvus_rag.db")
```
4. Vector databases work quite similarly to document databases like e.g. MongoDB. We define
   not a table, but a **collection** with specific **schema**, but conceptually it's a bit similar.
   For each element, we have an ID, text, and its embedding.
```python
from pymilvus import FieldSchema, DataType, CollectionSchema

VECTOR_LENGTH = 768  # check the dimensionality for Silver Retriever Base (v1.1) model

id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="Primary id")
text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096, description="Page text")
embedding_text = FieldSchema("embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_LENGTH, description="Embedded text")

fields = [id_field, text, embedding_text]

schema = CollectionSchema(fields=fields, auto_id=True, enable_dynamic_field=True, description="RAG Texts collection")
```
5. To create a collection with the given schema:
```python
COLLECTION_NAME = "rag_texts_and_embeddings"

milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema
)

index_params = milvus_client.prepare_index_params()

index_params.add_index(
    field_name="embedding", 
    index_type="HNSW",
    metric_type="L2",
    params={"M": 4, "efConstruction": 64}  # lower values for speed
) 

milvus_client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)

# checkout our collection
print(milvus_client.list_collections())

# describe our collection
print(milvus_client.describe_collection(COLLECTION_NAME))
```
6. Now we are able to insert documents into put database. RAG is the most useful when information
   is very specialized, niche, or otherwise probably unknown to the model or less popular. Let's
   start with ["IAB POLSKA Przewodnik po sztucznej inteligencji"](https://www.iab.org.pl/wp-content/uploads/2024/04/Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.pdf).
   This part is inspired by [SpeakLeash](https://www.speakleash.org/) and one of their projects
   [Bielik-how-to-start](https://github.com/speakleash/Bielik-how-to-start?tab=readme-ov-file) - 
   [Bielik_2_(4_bit)_RAG example](https://colab.research.google.com/drive/1ZdYsJxLVo9fW75uonXE5PCt8MBgvyktA?authuser=1). Bielik is the first Polish LLM, and you can also explore
   other tutorials for its usage. Let's define some constants for a start:
```python
# define data source and destination
## the document origin destination from which document will be downloaded 
pdf_url = "https://www.iab.org.pl/wp-content/uploads/2024/04/Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.pdf"

## local destination of the document
file_name = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.pdf"

## local destination of the processed document 
file_json = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.json"

## local destination of the embedded pages of the document
embeddings_json = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska-Embeddings.json"

## local destination of all above local required files
data_dir = "./data"
```
7. Let's download the document into the `data_dir` directory:
```python
# download data
import os
import requests

def download_pdf_data(pdf_url: str, file_name: str) -> None:
    response = requests.get(pdf_url, stream=True)
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, file_name), "wb") as file:
        for block in response.iter_content(chunk_size=1024):
            if block:
                file.write(block)

download_pdf_data(pdf_url, file_name)
```
8. This is a lot of text, and in RAG we need to add specific fragments to the prompt.
   To keep things simple, and number of vectors not too large, we will treat each page
   as a separate **chunk** to vectorize and search for. Below, we paginate document and save each
   page separately into a JSON file in format `{"page": page_number, "text": text_of_the_page}`.
```python
# prepare data

import fitz
import json


def extract_pdf_text(file_name, file_json):
    document = fitz.open(os.path.join(data_dir, file_name))
    pages = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = page.get_text()
        pages.append({"page_num": page_num, "text": page_text})

    with open(os.path.join(data_dir, file_json), "w") as file:
        json.dump(pages, file, indent=4, ensure_ascii=False)

extract_pdf_text(file_name, file_json)
```
9. Now we have texts, but we need vectors. We will use the model to embed text from each page
    and save the result in out collection in Milvus. It's very easy if we first prepare a single
    JSON file with all data. Its format is `{"page": page_num, "embedding": embedded_text}`.
```python
# vectorize data

import torch
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_embeddings(file_json, embeddings_json, model):
    pages = []
    with open(os.path.join(data_dir, file_json), "r") as file:
        data = json.load(file)

    for page in data:
        pages.append(page["text"])

    embeddings = model.encode(pages)

    embeddings_paginated = []
    for page_num in range(len(embeddings)):
        embeddings_paginated.append({"page_num": page_num, "embedding": embeddings[page_num].tolist()})

    with open(os.path.join(data_dir, embeddings_json), "w") as file:
        json.dump(embeddings_paginated, file, indent=4, ensure_ascii=False)

model_name = "ipipan/silver-retriever-base-v1.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
generate_embeddings(file_json, embeddings_json, model)
```
10. Now we can easily insert the data into Milvus:
```python
def insert_embeddings(file_json, embeddings_json, client=milvus_client):
    rows = []
    with open(os.path.join(data_dir, file_json), "r") as t_f, open(os.path.join(data_dir, embeddings_json), "r") as e_f:
        text_data, embedding_data = json.load(t_f), json.load(e_f)
        text_data =  list(map(lambda d: d["text"], text_data))
        embedding_data = list(map(lambda d: d["embedding"], embedding_data))
        
        for page, (text, embedding) in enumerate(zip(text_data, embedding_data)):
            rows.append({"text":text, "embedding": embedding})

    client.insert(collection_name="rag_texts_and_embeddings", data=rows)


insert_embeddings(file_json, embeddings_json)

# load inserted data into memory
milvus_client.load_collection("rag_texts_and_embeddings")
```

11. Now let's do some semantic search!
```python
# search

def search(model, query, client=milvus_client):
    embedded_query = model.encode(query).tolist()
    result = client.search(
        collection_name="rag_texts_and_embeddings", 
        data=[embedded_query], 
        limit=1,
        search_params={"metric_type": "L2"},
        output_fields=["text"]
    )
    return result


result = search(model, query="Czym jest sztuczna inteligencja")
```

However, **this is not yet RAG!**. This is just searching through our embeddings, without
any LLM or generation. Many companies rely on external LLMs used via API, due to easy setup,
good scalability, and low cost. We will follow this trend here and use Google Gemini API
to generate answer with RAG.

### Gemini API integration

Gemini API is free to use, with rate limit for the [free version of the API](https://ai.google.dev/pricing?hl=en).
We can use this LLM for intergration with our RAG system.

1. Get the API key from the Gemini API. Go to [model info](https://ai.google.dev/gemini-api/docs/pricing?hl=en#gemini-2.5-flash), 
   and click "Try it in Google AI Studio".
![Try it now](images/model_info.png)

2. You will be redirected to the Google AI Studio. Click "Create API Key", create a test project,
   and a test key.
![Get API Key](images/google_ai_studio.png)

3. Copy the key and save it in the environment variable. This is a **secret**, like any other API key, and **must never be shared**!
```bash
export GEMINI_API_KEY=your_api_key
```

4. Let's prepare the function that will call Google API and generate our response.
```python
import os
from google import genai

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_KEY)

MODEL = "gemini-2.5-flash"

def generate_response(prompt: str):
    try:
        # Send request to Gemini 2.5 Flash API and get the response
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        return response.text 
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
```

5. Now we can fully integrate everything into a RAG system. Fill the function below that will
   augment the prompt with knowledge from Milvus, and then use the LLM to generate an answer
   based on that context.
```python
def build_prompt(context: str, query: str) -> str:
    prompt = # try to build the prompt by yourself!
    return prompt
    

def rag(model, query: str) -> str:
    # having all prepared functions, you can combine them together and try to build your own RAG!
```

6. Test the RAG system with a few sample queries. Then experiment with the prompt - small wording changes can dramatically reshape the output. Try at least 2–3 of:
   - **Style:** force pirate speak, Shakespearean English, Yoda, or corporate jargon.
   - **Language:** instruct the model to answer in Polish (or Japanese, Latin, …)
     even though the retrieved context is in English.
   - **Format:** force a haiku, a single sentence, exactly three bullet points,
     or "emoji + one-line explanation" per fact.
   - **Persona:** a grumpy professor; an over-enthusiastic YouTuber; a skeptical
     reviewer that grades each claim "supported / unsupported by context".
   - **Constraints:** answer without using the letter "e"; max 50 words;
     each sentence starts with the next letter of the alphabet.
   - **Hallucination probe:** ask something *not* in the retrieved context and
     check whether the model invents an answer or honestly says "not in sources".
     A good RAG prompt should make refusal easy.

   Note which prompt tweaks survive retrieval (model stays grounded) and which
   make it drift away from the context. That tension is the core RAG craft.
