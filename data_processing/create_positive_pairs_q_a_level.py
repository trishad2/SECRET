#arg parse for q_a_level.py
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
import transformers
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from transformers import AdamW
import ast
from tqdm import tqdm
import numpy as np
import gc
import json
import numpy as np
import faiss
import warnings
warnings.filterwarnings("ignore")

# Initialize the parser
parser = argparse.ArgumentParser(description="create positive pairs for q_a_level.py")

# Add arguments
parser.add_argument('-f', '--file', type=str, help="training dataset")
parser.add_argument('-v', '--validation_x', type=str, help="validation dataset")

#example command: python3 create_positive_pairs_q_a_level.py -f '../data/demo_train_data.csv' -v '../data/val_data.csv'

# Parse arguments
args = parser.parse_args()

# Load the BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

#read the training dataset
file = args.file
#read 1000 rows of the training dataset , nrows=100
df = pd.read_csv(file)


val_file = args.validation_x
val_df = pd.read_csv(val_file)

#remove the rows from df if the nct_id is in val_df's nct_id
df = df[~df['nct_id'].isin(val_df['nct_id'])]

#reset the index
df.reset_index(drop=True, inplace=True)

print(df.shape)

#functions ---------------------------------------------------------------- start

def find_positive_samples(cosine_similarities, q_a_pairs):
    np.fill_diagonal(cosine_similarities, -np.inf)

    #find the index of the q/a pair with the highest cosine similarity
    max_similarities = []
    for i in range(len(cosine_similarities)):
        max_similarities.append(np.argmax(cosine_similarities[i]))

    #find the q/a pair with the highest cosine similarity
    max_pairs = []
    for i in range(len(q_a_pairs)):
        max_pairs.append(q_a_pairs[max_similarities[i]])

    return max_pairs

def find_positive_samples_faiss(indices, q_a_pairs):
    """
    Find positive samples for each query based on FAISS nearest neighbor indices.

    Args:
        indices (np.ndarray): Indices of nearest neighbors for each query, as returned by FAISS.
        q_a_pairs (list): The list of question/answer pairs.

    Returns:
        list: A list of positive samples corresponding to the nearest neighbor of each query.
    """
    positive_samples = []
    for i in range(len(indices)):
        # Use the first nearest neighbor (excluding itself)
        # FAISS search returns the query itself as the first neighbor (self-match), so we skip it
        nearest_neighbor_idx = indices[i][1]  # Second closest neighbor (index 1)
        positive_samples.append(q_a_pairs[nearest_neighbor_idx])
    return positive_samples


def calculate_cosine_similarity(embeddings):
    #first convert the embeddings to numpy arrays
    embeddings = [embedding.cpu().numpy().flatten() for embedding in embeddings]

    #calculate the cosine similarity between each pair of q/a pairs
    cosine_similarities = cosine_similarity(embeddings, embeddings)
    return cosine_similarities 

def calculate_cosine_similarity_chunked(embeddings, chunk_size=30000):
    embeddings = [embedding.cpu().numpy().flatten() for embedding in embeddings]
    n = len(embeddings)
    similarities = np.zeros((n, n))  # Consider sparse matrix if size is too large
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            chunk_sim = cosine_similarity(embeddings[i:i+chunk_size], embeddings[j:j+chunk_size])
            similarities[i:i+chunk_size, j:j+chunk_size] = chunk_sim
    return similarities

"""def calculate_ann_cosine_similarity(embeddings, top_k=10):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # L2 is equivalent to cosine similarity for normalized vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    distances, indices = index.search(embeddings, top_k)
    return distances, indices"""

def process_pairs(q_a_pairs, embed_text, find_positive_samples, file_handle):
    try:
        print(f"Processing {len(q_a_pairs)} pairs...")
        embeddings = []
        for pair in tqdm(q_a_pairs):
            embeddings.append(embed_text(pair))
        print("Embeddings computed.")

        cosine_similarities = calculate_cosine_similarity_chunked(embeddings)
        print("Cosine similarities calculated.")

        positive_samples = find_positive_samples(cosine_similarities, q_a_pairs)
        print("Positive samples identified.")

        for i in range(len(q_a_pairs)):
            json.dump({"anchor": q_a_pairs[i], "positive": positive_samples[i]}, file_handle)
            file_handle.write('\n')
        print("Data written to file.")

        del embeddings, cosine_similarities, positive_samples
        gc.collect()
    except Exception as e:
        print(f"Error occurred: {e}")
        raise




def process_pairs_faiss(q_a_pairs, embed_text, find_positive_samples, file_handle, top_k=2):
    try:
        print(f"Processing {len(q_a_pairs)} pairs...")

        # Step 1: Compute Embeddings
        embeddings = []
        for pair in tqdm(q_a_pairs):
            tensor_embedding = embed_text(pair)  # Assuming embed_text already does mean pooling
            # Ensure the tensor is moved to the CPU and converted to NumPy
            embeddings.append(tensor_embedding.cpu().numpy())  # Assuming embed_text returns a tensor

        embeddings = np.array(embeddings)  # Convert list of embeddings to NumPy array
        print(f"Embeddings shape before adding to FAISS: {embeddings.shape}")

        # Ensure embeddings is 2D (n_samples, embedding_dimension)
        if embeddings.ndim == 1:  # If embeddings is 1D, which shouldn't happen with BERT
            embeddings = embeddings.reshape(-1, 1)
        elif embeddings.ndim > 2:  # If there are more than 2 dimensions, flatten the extra dimensions
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        # Ensure the dtype is float32, as required by FAISS
        embeddings = embeddings.astype('float32')
        print("Embeddings computed.")

        # Step 2: Normalize Embeddings (for cosine similarity)
        faiss.normalize_L2(embeddings)

        # Step 3: Use FAISS for Nearest Neighbors Search
        print("Building FAISS index...")
        gpu_resource = faiss.StandardGpuResources()  # Initialize GPU resources
        d = embeddings.shape[1]  # Dimensionality of embeddings
        index = faiss.GpuIndexFlatIP(gpu_resource, d)  # Inner product for cosine similarity
        index.add(embeddings)  # Add embeddings to the index
        print("FAISS index built.")

        # Search for top-k neighbors for each embedding
        print("Searching for nearest neighbors...")
        distances, indices = index.search(embeddings, top_k)

        # Step 4: Generate Positive Samples
        print("Identifying positive samples...")
        positive_samples = find_positive_samples(indices, q_a_pairs)

        # Step 5: Write Results to File
        print("Writing data to file...")
        for i in range(len(q_a_pairs)):
            json.dump({"anchor": q_a_pairs[i], "positive": positive_samples[i]}, file_handle)
            file_handle.write('\n')

        # Cleanup
        print("Cleaning up...")
        del embeddings, distances, indices, positive_samples
        gc.collect()

        print("Processing complete.")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise



def cls_pooling(embeddings):
    return embeddings[:, 0, :] 

def embed_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings (hidden states from the last layer)
    # outputs.last_hidden_state -> (batch_size, sequence_length, hidden_size)
    embeddings = outputs.last_hidden_state

    # Pool the embeddings (e.g., by taking the mean across the sequence length)
    pooled_embeddings = cls_pooling(embeddings)

    return pooled_embeddings

#functions ---------------------------------------------------------------- end

df['q_a_criteria'] = df['q_a_criteria'].apply(ast.literal_eval)
df['q_a_intervention'] = df['q_a_intervention'].apply(ast.literal_eval)
df['q_a_disease'] = df['q_a_disease'].apply(ast.literal_eval)
df['q_a_outcome'] = df['q_a_outcome'].apply(ast.literal_eval)
df['q_a_keywords'] = df['q_a_keywords'].apply(ast.literal_eval)
df['q_a_title'] = df['q_a_title'].apply(ast.literal_eval)
#if q_a_description in columns
if 'q_a_description' in df.columns:
    df['q_a_description'] = df['q_a_description'].apply(ast.literal_eval)


print('Creating contrastive samples.........')
q_a_pairs_criteria = []
for row in df['q_a_criteria']:
    for pair in row:
        #if "Not Available" is not substring of pair
        if "Not Available" not in pair:
            q_a_pairs_criteria.append(pair)

q_a_pairs_intervention = []
for row in df['q_a_intervention']:
    for pair in row:
        if "Not Available" not in pair:
            q_a_pairs_intervention.append(pair)

q_a_pairs_disease = []
for row in df['q_a_disease']:
    for pair in row:
        if "Not Available" not in pair:
            q_a_pairs_disease.append(pair)

q_a_pairs_keywords = []
for row in df['q_a_keywords']:
    for pair in row:
        if "Not Available" not in pair:
            q_a_pairs_keywords.append(pair)

q_a_pairs_title = []
for row in df['q_a_title']:
    for pair in row:
        if "Not Available" not in pair:
            q_a_pairs_title.append(pair)

q_a_pairs_outcome = []
for row in df['q_a_outcome']:
    for pair in row:
        if "Not Available" not in pair:
            q_a_pairs_outcome.append(pair)

q_a_pairs_description = []
if 'q_a_description' in df.columns:
    for row in df['q_a_description']:
        for pair in row:
            if "Not Available" not in pair:
                q_a_pairs_description.append(pair)

#change the file name accordingly

with open(f'../data/text_q_a_level_demo.json', 'w') as f:
        """process_pairs(q_a_pairs_criteria, embed_text, find_positive_samples, f)
        process_pairs(q_a_pairs_intervention, embed_text, find_positive_samples, f)
        process_pairs(q_a_pairs_disease, embed_text, find_positive_samples, f)
        process_pairs(q_a_pairs_keywords, embed_text, find_positive_samples, f)
        process_pairs(q_a_pairs_title, embed_text, find_positive_samples, f)
        process_pairs(q_a_pairs_outcome, embed_text, find_positive_samples, f)"""

        process_pairs_faiss(q_a_pairs_criteria, embed_text, find_positive_samples_faiss, f)
        process_pairs_faiss(q_a_pairs_intervention, embed_text, find_positive_samples_faiss, f)
        process_pairs_faiss(q_a_pairs_disease, embed_text, find_positive_samples_faiss, f)
        process_pairs_faiss(q_a_pairs_keywords, embed_text, find_positive_samples_faiss, f)
        process_pairs_faiss(q_a_pairs_title, embed_text, find_positive_samples_faiss, f)
        process_pairs_faiss(q_a_pairs_outcome, embed_text, find_positive_samples_faiss, f)
        if 'q_a_description' in df.columns:
            process_pairs_faiss(q_a_pairs_description, embed_text, find_positive_samples_faiss, f)
