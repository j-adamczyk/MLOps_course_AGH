# Laboratory 8 instruction

## Introduction

This lab will cover vector search, vector databases, and their applications like similarity search
and Retrieval-Augmented Generation (RAG). We will work with two major approaches to implementing
this in MLOps practice: vector database (Milvus) and vector index (Postgres extension).

## PostgreSQL + TimescaleDB + pgvectorscale (2 points)

### Prebuilt Docker image

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

### Preparing vector database with Docker Compose

1. Create `vectorscale_db` directory and go there. We will prepare Docker Compose there.
2. Postgres Docker containers can run SQL script upon initialization, which is very useful
   for installing and enabling extensions. Create `initdb` directory, and
   `create_similar_search_service_db.sql` file inside. Fill its contents:
```sql
-- prepare_similarity_search_service_db.sql

-- Create the database for the similarity search service
# CREATE DATABASE similarity_search_service_db;

-- Connect to the database
...

-- Enable vectorscale extension
...
```
3. Create `docker-compose.yml` file, using provided template. It will create
   the `timescaledb` service with TimescaleDB container and mount the `initdb`
   directory to the `/docker-entrypoint-initdb.d` directory in the container. 
   This directory is used to initialize any SQL scripts when the Postgres server is started.
   Fill the template as necessary.
```yaml
version: '3.9'

services:
  timescaledb:
    # define image
    container_name: vectorscaledb
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_USER: postgres
    ports:
      - "xxxx:yyyy"
    volumes:
      - # map the initdb prepare db script to :/docker-entrypoint-initdb.d/prepare_similarity_search_service_db.sql
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
    port=xxxx,
    database="similarity_search_service_db"
)
```

Tables in SQLAlchemy are defined using class-based design. Typically, application defines
a single `Base` class, from which concrete tables inherit. They are defined quite similarly
to Pydantic, i.e. with attributes and types. However, here we also need to assign concrete
database types. To integrate it with vector search, we will also use `pgvector` library.

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, List
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
import numpy as np

# reusable function to insert data into the table
def insert_image(engine: sqlalchemy.Engine, image_path: str, image_embedding: list[float]):
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


# calculate the cosine similarity between the first image and the K rest of the images, order the images by the similarity score
def find_k_images(engine: sqlalchemy.Engine, k: int, orginal_image: Image) -> list[Image]:
    with Session(engine) as session:
        # execution_options={"prebuffer_rows": True} is used to prebuffer the rows, this is useful when we want to fetch the rows in chunks and return them after session is closed
        result = session.execute(
            select(Image)
            .order_by(Image.image_embedding.cosine_similarity(orginal_image.image_embedding))
            .limit(k), 
            execution_options={"prebuffer_rows": True}
        )
        return result

# find the 10 most similar images to the first image
k = 10
similar_images = find_k_images(engine, k, image)
```

### Filtering

We can filter the found nearest neighbors, e.g. by required minimal similarity score.

```python
# find the images with the similarity score greater than 0.9
def find_images_with_similarity_score_greater_than(engine: sqlalchemy.Engine, similarity_score: float, orginal_image: Image) -> list[Image]:
    with Session(engine) as session:
        result = session.execute(
            select(Image)
            .filter(Image.image_embedding.cosine_similarity(orginal_image.image_embedding) > similarity_score), 
            execution_options={"prebuffer_rows": True}
        )
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

columns_to_keep = ["Name", "Windows", "Linux", "Mac", "About the game", "Supported languages", "Price"]

N = 40000
dataset = dataset["train"].select_columns(columns_to_keep).select(range(N))
```

We will use columns:
* `Name`
* `About the game`
* `Price`
* `Platforms` - the platforms on which the game is available; note that there is separate field
  for each platform (Windows, Linux, macOS)

For vector search, we can use the `About the game` column, which is an arbitrary text description. A great model
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
from sqlalchemy import Integer, Float, Boolean


class Games(Base):
    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}
    
    # the vector size produced by the model taken from documentation https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    VECTOR_LENGTH = # check the model output dimensionality 
        
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(String(4096))
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
        game_description = game["About the game"] or ""
        game_embedding = generate_embeddings(game_description)
        name, windows, linux, mac, price = game["Name"], game["Windows"], game["Linux"], game["Mac"], game["Price"]
        if name and windows and linux and mac and price and game_description:
            game = Games(
                name=game["Name"], 
                description=game_description[0:4096],
                windows=game["Windows"], 
                linux=game["Linux"], 
                mac=game["Mac"], 
                price=game["Price"], 
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
```python
def find_game(
    engine: sqlalchemy.Engine, 
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

In such cases, we don't relaly need any additional capabilities like attributes filtering, ACID,
JOINs or other Postgres-related advantages. Thus, we will use [Milvus](https://milvus.io/), a typical
example of vector database. To generate embeddings, we will use
[Silver Retriever model](https://huggingface.co/ipipan/silver-retriever-base-v1.1) from Sentence Transformers.
It is based on HerBERT model for Polish language, and finetuned for retrieval of similar vectors.

1. Start by setting up Milvus by using its Docker image. Docker Compose file is also conveniently
   provided by its creators:
```bash
mkdir milvus_db
cd milvus_db

wget https://github.com/milvus-io/milvus/releases/download/v2.4.13-hotfix/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
2. Run the database with `docker compose up -d`.
3. Next code sections are quite interactive and will probably be easier to run inside a Jupyter
   Notebook. Start it with `jupyter notebook`.
4. Let's connect to the database. Milvus provides its own `pymilvus` library.
```python
from pymilvus import MilvusClient

host = "localhost"
port = "19530"

milvus_client = MilvusClient(
    host=host,
    port=port
)
```
5. Vector databases work quite similarly to document databases like e.g. MongoDB. We define
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
6. To create a collection with the given schema:
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
7. Now we are able to insert documents into put database. RAG is the most useful when information
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
8. Let's download the document into the `data_dir` directory:
```python
# download data
import os
import requests

def download_pdf_data(pdf_url: str, file_name: str) -> None:
    response = requests.get(pdf_url, stream=True)
    with open(os.path.join(data_dir, file_name), "wb") as file:
        for block in response.iter_content(chunk_size=1024):
            if block:
                file.write(block)

download_pdf_data(pdf_url, file_name)
```
9. This is a lot of text, and in RAG we need to add specific fragments to the prompt.
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
10. Now we have texts, but we need vectors. We will use the model to embed text from each page
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
11. Now we can easily insert the data into Milvus:
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

12. Now let's do some semantic search!
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

MODEL = "gemini-2.0-flash"

def generate_response(prompt: str):
    try:
        # Send request to Gemini 2.0 Flash API and get the response
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

6. Test the RAG system with a few sample queries.
