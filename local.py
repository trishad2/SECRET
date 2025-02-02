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
from info_nce import InfoNCE, info_nce
from torch.utils.data import DataLoader
from transformers import AdamW
import ast
from tqdm import tqdm
import numpy as np
import gc
import json
import warnings
warnings.filterwarnings("ignore")

# Initialize the parser
parser = argparse.ArgumentParser(description="Q_a level training")

# Add arguments
#parser.add_argument('-f', '--file', type=str, help="training dataset")
parser.add_argument('-p', '--contrastive', type=str, help="contrastive pairs")
parser.add_argument('-v', '--validation_x', type=str, help="validation dataset")
parser.add_argument('-l', '--validation_y', type=str, help="validation labels")
parser.add_argument('-a', '--autocast', type=bool, help="autocast")
parser.add_argument('-m', '--metric', type=str, help="metric to evaluate the model")


#example: CUDA_VISIBLE_DEVICES=0,1 python3 local.py  -p 'data/text_q_a_level_demo.json' -v 'data/val_data.csv' -l 'data/val_list.csv' -a True -m 'recall@5'

# Parse arguments
args = parser.parse_args()

val_metric = args.metric

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



val_file = args.validation_x
val_df = pd.read_csv(val_file)

val_trial_list = args.validation_y
val_trial_list_df = pd.read_csv(val_trial_list)

val_trial_list_df_rank = val_trial_list_df[['nct_id', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5', 'rank_6', 'rank_7', 'rank_8', 'rank_9', 'rank_10']]
val_trial_list_df_truth = val_trial_list_df[['nct_id', 'truth_1', 'truth_2', 'truth_3', 'truth_4', 'truth_5', 'truth_6', 'truth_7', 'truth_8', 'truth_9', 'truth_10']]


contrastive_samples = args.contrastive

#functions.........................................start
def find_most_similar_q_a_pair(q_a_pair, tfidf_matrix, tfidf, q_a_pairs):
    q_a_pair_vector = tfidf.transform([q_a_pair])
    cosine_similarities = cosine_similarity(q_a_pair_vector, tfidf_matrix).flatten()
    most_similar_index = cosine_similarities.argsort()[-2]
    most_similar_q_a_pair = q_a_pairs[most_similar_index]
    return most_similar_q_a_pair

def tokenize_pairs(batch):
    anchor_tokens = tokenizer(
        batch["anchor"], padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    positive_tokens = tokenizer(
        batch["positive"], padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    """negative_tokens = tokenizer(
        batch["negative"], padding=True, truncation=True, return_tensors="pt", max_length=512
    )"""
    return anchor_tokens, positive_tokens#, negative_tokens

def cls_pooling(embeddings):
    return embeddings[:, 0, :] 

def create_single_list(row):
    single_list = []
    for q_a_pairs in [row['q_a_criteria'], row['q_a_intervention'], row['q_a_disease'],  row['q_a_keywords'], row['q_a_title'], row['q_a_outcome']]:
        for pair in q_a_pairs:
            single_list.append(pair)
    return single_list
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

def embed_text1(text):
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

def precision(ranked_label_list, k):
    return np.mean(ranked_label_list[:,:k].sum(1) / k)

def recall(ranked_label_list, k):
    return (ranked_label_list[:,:k].sum(1) / ranked_label_list.sum(1)).mean()

def ndcg(ranked_label_list, k=None):
    if k is None: k = ranked_label_list.shape[1]
    discount = np.log2(np.arange(2,2+k))
    dcg = np.sum(ranked_label_list[:,:k] / discount, 1)
    idcg = np.sum(np.flip(np.sort(ranked_label_list.copy()), 1)[:,:k] / discount,1)
    ndcg = (dcg / idcg)
    return np.mean(ndcg)


def MAP(ranked_label_list):
    ap_scores = []
    for row in ranked_label_list:
        precisions = []
        num_relevant = 0
        for k in range(1, len(row) + 1):
            if row[k - 1] == 1:  # Check if item at rank k is relevant
                num_relevant += 1
                precisions.append(num_relevant / k)  # Precision@k
        if num_relevant > 0:
            ap_scores.append(np.mean(precisions))
        else:
            ap_scores.append(0)  # No relevant items
    return np.mean(ap_scores)

def evaluate(rank_data, truth_data, test_df):
    ranked_label_list = []

    for idx, row in rank_data.iterrows():
        target_trial = row['nct_id']
        target_emb = test_df[test_df['nct_id'] == target_trial][f'q_a_embs'].values[0].clone().detach().cpu().numpy()
        
        
        candidate_trials = row[1:].values.tolist()
        candidate_embs_list = []
        for candidate in candidate_trials:
            if candidate in test_df['nct_id'].values:
                emb = torch.tensor(test_df[test_df['nct_id'] == candidate]['q_a_embs'].values[0]).detach().cpu().numpy()
                candidate_embs_list.append(np.array(emb))
        candidate_embs = np.array([x.reshape(-1) for x in candidate_embs_list])
        
        # Calculate cosine similarity
        sim = cosine_similarity(target_emb.reshape(1, -1), candidate_embs)[0]

        labels = truth_data.iloc[idx, 1:].to_numpy()
        if labels.sum() == 0:
            continue

        ranked_label = labels[np.argsort(sim)[::-1]]
        ranked_label_list.append(ranked_label)

    ranked_label_list = np.array(ranked_label_list)

    return {
        f'precision@{k}': precision(ranked_label_list, k) for k in [1, 2, 5]
    } | {
        f'recall@{k}': recall(ranked_label_list, k) for k in [1, 2, 5]
    } | {
        f'ndcg@{k}': ndcg(ranked_label_list, k) for k in [1, 2, 5]
    } | {
        'MAP': MAP(ranked_label_list)
    }

def calculate_cosine_similarity(embeddings):
    #first convert the embeddings to numpy arrays
    embeddings = [embedding.cpu().numpy().flatten() for embedding in embeddings]

    #calculate the cosine similarity between each pair of q/a pairs
    cosine_similarities = cosine_similarity(embeddings, embeddings)
    return cosine_similarities 

def find_negative_samples(cosine_similarities, q_a_pairs):
    #min similarity that is not -np.inf
    min_similarities = []
    for i in range(len(cosine_similarities)):
        min_similarities.append(np.argmin(cosine_similarities[i]))

    #find the q/a pair with the lowest cosine similarity
    min_pairs = []
    for i in range(len(q_a_pairs)):
        min_pairs.append(q_a_pairs[min_similarities[i]])

    return min_pairs

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

def process_pairs(q_a_pairs, embed_text, find_positive_samples, file_handle):
    embeddings = []
    for pair in tqdm(q_a_pairs):
        embeddings.append(embed_text(pair))
    cosine_similarities = calculate_cosine_similarity(embeddings)
    positive_samples = find_positive_samples(cosine_similarities, q_a_pairs)
    
    for i in range(len(q_a_pairs)):
        json.dump({"anchor": q_a_pairs[i], "positive": positive_samples[i]}, file_handle)
        file_handle.write('\n')
    del embeddings, cosine_similarities, positive_samples
    gc.collect()
#functions.........................................end



val_df['q_a_criteria'] = val_df['q_a_criteria'].apply(ast.literal_eval)
val_df['q_a_criteria'] = val_df['q_a_criteria'].apply(lambda x: x[:10] if len(x) > 10 else x)
val_df['q_a_intervention'] = val_df['q_a_intervention'].apply(ast.literal_eval)
val_df['q_a_disease'] = val_df['q_a_disease'].apply(ast.literal_eval)
val_df['q_a_outcome'] = val_df['q_a_outcome'].apply(ast.literal_eval)
val_df['q_a_keywords'] = val_df['q_a_keywords'].apply(ast.literal_eval)
val_df['q_a_title'] = val_df['q_a_title'].apply(ast.literal_eval)
val_df['q_a'] = val_df.apply(create_single_list, axis=1)

val_df['q_a'] = val_df['q_a'].apply(lambda x: ". ".join(x))


with open(f'{contrastive_samples}', 'r') as file:
    texts = []
    for line in file:
        texts.append(json.loads(line))


loss = InfoNCE()
batch_size, embedding_size = 32, 768
accumulation_steps = 4  # Number of steps to accumulate gradients before optimizer step

# Dataloader for your dataset
dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Scaler for mixed precision training
scaler = GradScaler()

# Move model to the same device
model = model.to(device)

best_map = 0
best_model = model
best_epoch = 0

# Training loop
model.train()
print("Training loop")
for epoch in range(10):  # Number of epochs
    for step, batch in enumerate(dataloader):
        # Tokenize the batch
        anchor_tokens, positive_tokens = tokenize_pairs(batch)

        # Move tensors to GPU (if available)
        anchor_tokens = {k: v.to(device) for k, v in anchor_tokens.items()}
        positive_tokens = {k: v.to(device) for k, v in positive_tokens.items()}

        if bool(args.autocast):
            with autocast():
                # Extract embeddings
                anchor_emb = model(**anchor_tokens).last_hidden_state
                positive_emb = model(**positive_tokens).last_hidden_state

                # Pool the embeddings (CLS token or mean pooling)
                query = cls_pooling(anchor_emb)
                positive_key = cls_pooling(positive_emb)

                # Compute InfoNCE loss
                output = loss(query, positive_key)

            scaler.scale(output / accumulation_steps).backward()
        else:
            # Extract embeddings
            anchor_emb = model(**anchor_tokens).last_hidden_state
            positive_emb = model(**positive_tokens).last_hidden_state

            # Pool the embeddings (CLS token or mean pooling)
            query = cls_pooling(anchor_emb)
            positive_key = cls_pooling(positive_emb)

            # Compute InfoNCE loss
            output = loss(query, positive_key)

            # Backpropagation
            (output / accumulation_steps).backward()

        # Perform optimizer step and scaler update every `accumulation_steps`
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            if bool(args.autocast):
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        if (step + 1) % accumulation_steps == 0:
            print(f"Epoch {epoch}, Step {step + 1}, Loss: {output.item()}")

    tqdm.pandas()
    val_df['q_a_embs'] = val_df['q_a'].progress_apply(embed_text1)

    # Evaluate the model
    map = evaluate(val_trial_list_df_rank, val_trial_list_df_truth, val_df)[val_metric]

    if map > best_map:
        best_map = map
        best_model = model
        best_epoch = epoch

print(f"Best MAP: {best_map} at epoch {best_epoch}")

# Save the model
torch.save(best_model.module.state_dict(), 'local_model.pth')
