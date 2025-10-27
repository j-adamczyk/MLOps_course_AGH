## Laboratory 8 homework

Homework is creating a multimodal similarity search service. It will retrieve images
based on a text query, using embeddings from a multimodal model.

### Dataset preparation (1 point)

1. Download the dataset: [Amazon Berkeley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar)
2. Unpack the dataset and locate the metadata file `images.csv.gz` in `images/metadata/`
3. Extract the image paths from the dataset:
   - write a Python script that extracts valid image paths from the CSV file
   - keep only images with at least 1000 pixels of width and height
   - you can use Pandas, Polars etc. as you wish, add with `uv` whatever is necessary

### Model Selection and Understanding (1 point)

1. Read the [CLIP Image Search Tutorial](https://www.sbert.net/examples/applications/image-search/README.html).
2. Search for `clip-ViT-B-32` model from `sentence-transformers` on HuggingFace Hub.
3. Determine the vector length (size) produced by the model, and what type of model is it.

**Questions:**
- What is the vector size produced by `clip-ViT-B-32`?
- What do `ViT`, `B` and `32` mean in the model name?

### Database setup (2 points)

1. Set up Postgres database with `pgvectorscale` extension. You can reuse the database from
   the lab.
2. Use SQLAlchemy to connect to it with `create_engine` and `URL`.
3. Create a database table definition for storing image embeddings. Complete the following SQLAlchemy model:

```python
class Img(Base):
    __tablename__ = "images"
    __table_args__ = {'extend_existing': True}
    
    VECTOR_LENGTH: int = ... # implement me!
    
    id: Mapped[int] = mapped_column(primary_key=True)
    image_path: = ... # implement me!
    embedding: =  ... # implement me! 
    
# Create table
Base.metadata....
```

### Image vectorization (3 points)

Fill the code below that performs image vectorization, i.e. calculates an embedding for each
image. Model input is an image, its output is a vector (embedding).

Notes:
1. Adjust `MAX_IMAGES`, the number of images processed, based on your computational capabilities.
   The more images you process, the more accureate results of search you will get. If you have a
   GPU, you can use it and process more images.
2. Load the CLIP model with `sentence-transformers`. It is a nice, high-level framework, based on
   PyTorch. To enable GPU usage, the following code is useful:
```python
device = "cuda" if torch.cuda.is_available() else "cpu" 
```
3. Implement [batching](https://docs.python.org/3/library/itertools.html#itertools.batched) for efficiency,
   as embedding a single image at a time is very inefficient. In that case, you would spend most of the time
   sending data between RAM and CPU/GPU, instead of actually doing the work. Working in batches optimizes
   this transfer time. Similarly, batching is more efficient for databases, as it lowers the overhead for
   managing transactions.
4. Implement all necessary code parts:
   - iterate through image paths
   - implement batching
   - insert batch of images into the database
   - update progress bar `pgar` with appropriate batch size

```python
import joblib
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

MAX_IMAGES = # Adjust it for your needs
BATCH_SIZE = joblib.cpu_count(only_physical_cores=True)

def insert_images(engine, images):
    # finish

def vectorize_images(engine, model, image_paths):    
    with tqdm(total=MAX_IMAGES) as pbar:
        for images_paths_batch in ...:  # finish (suggestion - use `batched`)
            images = [Image.open(path) for path in images_paths_batch]
        
            # calculate embeddings
        
            # create Img instances for all images in batch
        
            # insert all batch images
        
            # update pbar
            pbar.update(len(imgs))

vectorize_images(engine, model, image_paths)
```

### Search and results display (3 points)

Fill the code below that searches for the most similar embeddings based
on a text description. After that:
1. Check a few different queries. Are results relevant?
2. Check 3 different MAX_IMAGES sizes. Do you see more accurate results for larger datasets?

```python
import matplotlib.pyplot as plt

class ImageSearch:
    def __init__(self, engine, model):
        self.engine = engine
        self.model = model
        
    def __call__(self, image_description: str, k: int):
        found_images = self.find_similar_images(image_description, k)
        # display images

    def find_similar_images(self, image_description: str, k: int):
        image_embedding = ... # calculate embedding of image_description
        
        # remember about session and commit
        query = (
            # write query to find K images with highest cosine distance
        )
        result = ... # execute query
        return result
    
    def display_images(self, images):
        fig, axes = plt.subplots(1, k, figsize=(15, 5))
        
        for i, img_path in enumerate(images):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i+1}")
        
        plt.show()
```
