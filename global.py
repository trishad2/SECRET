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
import json
import numpy as np
import warnings
import pickle
from info_nce import InfoNCE, info_nce
from collections import OrderedDict



warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description= "trial level semisupervised learning")

parser.add_argument('-f', '--file', type=str, help="json dataset")
parser.add_argument('-v', '--validation_x', type=str, help="validation dataset")
parser.add_argument('-l', '--label', type=str, help="validation labels")
parser.add_argument('-a', '--autocast', type= bool, help="use autocast for mixed precision training")
parser.add_argument('-m', '--metric', type=str, help="metric to evaluate the model")


#example: CUDA_VISIBLE_DEVICES=4,7 python3 global.py -f data/text_trial_level_demo.json -v data/val_data.csv -l data/val_list.csv -a True -m recall@5

args = parser.parse_args()

val_metric = args.metric   #metric to evaluate the model

# Load the BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Model loaded from q_a level")
state_dict = torch.load('local.pth')


new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

#functions ---------------------------------------------------------------- start
def tokenize_pairs(batch):
    anchor_tokens = tokenizer(
        batch["anchor"], padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    positive_tokens = tokenizer(
        batch["positive"], padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    return anchor_tokens, positive_tokens

def cls_pooling(embeddings):
    return embeddings[:, 0, :] 

def create_single_list(row):
    single_list = []
    for q_a_pairs in [row['q_a_criteria'], row['q_a_intervention'], row['q_a_disease'],  row['q_a_keywords'], row['q_a_title'], row['q_a_outcome']]:
        for pair in q_a_pairs:
            single_list.append(pair)
    return single_list

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
def MAP(ranked_label_list):
    return np.mean([np.mean([precision(ranked_label_list, k) for k in range(1,ranked_label_list.shape[1]+1)])])

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

def custom_collate(batch):
    anchor_texts = [item['anchor'] for item in batch]
    positive_texts = [item['positive'] for item in batch]
    negative_texts = [item['negative'] for item in batch]

    # Example: Convert list of tokens to a single string
    anchor_texts = [". ".join(tokens) for tokens in anchor_texts]
    positive_texts = [". ".join(tokens) for tokens in positive_texts]
    negative_texts = [". ".join(tokens) for tokens in negative_texts]

    # Tokenize the texts
    anchor_tokens = tokenizer(anchor_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    positive_tokens = tokenizer(positive_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    negative_tokens = tokenizer(negative_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return anchor_tokens, positive_tokens, negative_tokens
#functions ---------------------------------------------------------------- end

# Load the dataset
with open(args.file) as f:
    texts = json.load(f)

val_df = pd.read_csv(args.validation_x)

val_df['q_a_criteria'] = val_df['q_a_criteria'].apply(ast.literal_eval)
val_df['q_a_criteria'] = val_df['q_a_criteria'].apply(lambda x: x[:10] if len(x) > 10 else x)
val_df['q_a_intervention'] = val_df['q_a_intervention'].apply(ast.literal_eval)
val_df['q_a_disease'] = val_df['q_a_disease'].apply(ast.literal_eval)
val_df['q_a_outcome'] = val_df['q_a_outcome'].apply(ast.literal_eval)
val_df['q_a_keywords'] = val_df['q_a_keywords'].apply(ast.literal_eval)
val_df['q_a_title'] = val_df['q_a_title'].apply(ast.literal_eval)
val_df['q_a'] = val_df.apply(create_single_list, axis=1)

val_df['q_a'] = val_df['q_a'].apply(lambda x: ". ".join(x))

val_trial_list = args.label
val_trial_list_df = pd.read_csv(val_trial_list)

val_trial_list_df_rank = val_trial_list_df[['nct_id', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5', 'rank_6', 'rank_7', 'rank_8', 'rank_9', 'rank_10']]
val_trial_list_df_truth = val_trial_list_df[['nct_id', 'truth_1', 'truth_2', 'truth_3', 'truth_4', 'truth_5', 'truth_6', 'truth_7', 'truth_8', 'truth_9', 'truth_10']]


loss = InfoNCE(negative_mode='paired')
loss1 = InfoNCE()
batch_size, embedding_size = 16, 768


# Dataloader for your dataset
dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-6)

#scaler for mixed precision training
scaler = GradScaler()

# Move model to the same device
model = model.to(device)

best_map = 0
best_model = model
best_epoch = 0

# Training loop
model.train()
for epoch in range(10):  # Number of epochs
    for anchor_tokens, positive_tokens, negative_tokens in dataloader:
        # Move tensors to GPU (if available)
        anchor_tokens = {k: v.to(device) for k, v in anchor_tokens.items()}
        positive_tokens = {k: v.to(device) for k, v in positive_tokens.items()}
        negative_tokens = {k: v.to(device) for k, v in negative_tokens.items()}

        if bool(args.autocast):
            #use autocast to reduce the memory usage
            with autocast():
                # Extract embeddings
                anchor_emb = model(**anchor_tokens).last_hidden_state
                positive_emb = model(**positive_tokens).last_hidden_state
                negative_emb = model(**negative_tokens).last_hidden_state

                # Pool the embeddings (CLS token or mean pooling)
                query = cls_pooling(anchor_emb)
                positive_key = cls_pooling(positive_emb)
                negative_key = cls_pooling(negative_emb)

                #negative_key to shape (batch_size, negative_samples, embedding_size)
                negative_key = negative_key.unsqueeze(1).repeat(1, 2, 1)
                
                paired_info_nce = loss(query, positive_key, negative_key)
                batch_info_nce = loss1(query, positive_key)

                print(f"Paired InfoNCE: {paired_info_nce.item()}, Batch InfoNCE: {batch_info_nce.item()}")
                # Compute InfoNCE loss
                output = paired_info_nce + batch_info_nce
                #loss(query, positive_key, negative_key) + loss1(query, positive_key)

            optimizer.zero_grad()
            # Backpropagation # Backpropagation with scaled gradients
            scaler.scale(output).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Extract embeddings
            anchor_emb = model(**anchor_tokens).last_hidden_state
            positive_emb = model(**positive_tokens).last_hidden_state
            negative_emb = model(**negative_tokens).last_hidden_state

            # Pool the embeddings (CLS token or mean pooling)
            query = cls_pooling(anchor_emb)
            positive_key = cls_pooling(positive_emb)
            negative_key = cls_pooling(negative_emb)

            #negative_key to shape (batch_size, negative_samples, embedding_size)
            negative_key = negative_key.unsqueeze(1).repeat(1, 2, 1)

            paired_info_nce = loss(query, positive_key, negative_key)
            batch_info_nce = loss1(query, positive_key)

            print(f"Paired InfoNCE: {paired_info_nce.item()}, Batch InfoNCE: {batch_info_nce.item()}")
            # Compute InfoNCE loss
            output = paired_info_nce + batch_info_nce

            # Backpropagation
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
        

        print(f"Epoch {epoch}, Loss: {output.item()}")

    tqdm.pandas()  
    #use the model to embed df_val['q_a']
    val_df['q_a_embs'] = val_df['q_a'].progress_apply(embed_text1)

    #evaluate the model
    map = evaluate(val_trial_list_df_rank, val_trial_list_df_truth, val_df)[val_metric]

    if map > best_map:
        best_map = map
        best_model = model
        best_epoch = epoch 
        print(f"Best MAP: {best_map} at epoch {epoch}")

        


print(f"Best MAP: {best_map} at best epoch {best_epoch}")

# Save the model
torch.save(best_model.module.state_dict(), 'global_model.pth')
