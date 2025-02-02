from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pandas as pd
import ast
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

state_dict = torch.load('models/global_model.pth')

# Remove `module.` prefix if present
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()


test_df = pd.read_csv('data/test_data.csv')

n = 10
test_df['q_a_criteria'] = test_df['q_a_criteria'].apply(ast.literal_eval)
test_df['q_a_criteria'] = test_df['q_a_criteria'].apply(lambda x: x[:n] if len(x) > n else x)

test_df['q_a_intervention'] = test_df['q_a_intervention'].apply(ast.literal_eval)
test_df['q_a_disease'] = test_df['q_a_disease'].apply(ast.literal_eval)
test_df['q_a_outcome'] = test_df['q_a_outcome'].apply(ast.literal_eval)
test_df['q_a_keywords'] = test_df['q_a_keywords'].apply(ast.literal_eval)
test_df['q_a_title'] = test_df['q_a_title'].apply(ast.literal_eval)


def create_single_list(row):
    single_list = []
    for q_a_pairs in [row['q_a_criteria'], row['q_a_intervention'], row['q_a_disease'],  row['q_a_keywords'], row['q_a_title'], row['q_a_outcome']]:
        for pair in q_a_pairs:
            single_list.append(pair)
    return single_list

test_df['q_a'] = test_df.apply(create_single_list, axis=1)



def embed_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings (hidden states from the last layer)
    # outputs.last_hidden_state -> (batch_size, sequence_length, hidden_size)
    embeddings = outputs.last_hidden_state.to(device)

    # Pool the embeddings (e.g., by taking the mean across the sequence length)
    pooled_embeddings = embeddings.mean(dim=1)

    return pooled_embeddings

#convert the q_a to a string
test_df['q_a'] = test_df['q_a'].apply(lambda x: ". ".join(x))

# Embed the Q-A pairs
tqdm.pandas()  # Initialize tqdm for progress_apply
test_df['q_a_embs'] = test_df['q_a'].progress_apply(embed_text)

#test_trial_list_df = pd.read_excel('/home/trishad2/trial_searching/data/TrialSim-data.xlsx', index_col=0)
test_trial_list_df = pd.read_csv('data/test_list.csv')

#combined test
#test_trial_list_df = pd.read_csv('/home/trishad2/trial_searching/data/test_list_combined.csv')

test_df = test_df[['nct_id', 'q_a_embs']]
test_df = test_df.reset_index(drop=True)

test_df['q_a_embs'] = test_df['q_a_embs'].apply(lambda x: x.cpu().numpy())

test_trial_list_df = test_trial_list_df.rename(columns={'target_trial':'nct_id',1: 'truth_1', 2: 'truth_2', 3: 'truth_3', 4: 'truth_4', 5: 'truth_5', 6: 'truth_6', 7: 'truth_7', 8: 'truth_8', 9: 'truth_9', 10: 'truth_10'})

test_trial_list_df_rank = test_trial_list_df[['nct_id', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5', 'rank_6', 'rank_7', 'rank_8', 'rank_9', 'rank_10']]
test_trial_list_df_truth = test_trial_list_df[['nct_id', 'truth_1', 'truth_2', 'truth_3', 'truth_4', 'truth_5', 'truth_6', 'truth_7', 'truth_8', 'truth_9', 'truth_10']]


def precision(ranked_label_list, k):
    return np.mean(ranked_label_list[:,:k].sum(1) / k)

def recall(ranked_label_list, k):
    return (ranked_label_list[:,:k].sum(1) / ranked_label_list.sum(1)).mean()

def ndcg(ranked_label_list, k=None):
    if k is None: k = ranked_label_list.shape[1]
    discount = np.log2(np.arange(2,2+k))
    dcg = np.sum(ranked_label_list[:,:k] / discount, 1)
    idcg = np.sum(np.flip(np.sort(ranked_label_list.copy()), 1)[:,:k] / discount,1)
    ndcg = (dcg / idcg).mean()
    return ndcg

"""def MAP(ranked_label_list):
    return np.mean([np.mean([precision(ranked_label_list, k) for k in range(1,ranked_label_list.shape[1]+1)])])"""

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
    
    # Set test_df index for faster lookups
    test_df = test_df.set_index('nct_id')
    
    for idx, row in rank_data.iterrows():
        target_trial = row['nct_id']
        target_emb = test_df.loc[target_trial, 'q_a_embs']
        
        # Get candidate trials and their embeddings
        candidate_trials = row[1:].values.tolist()
        candidate_embs = test_df.loc[candidate_trials, 'q_a_embs'].tolist()
        candidate_embs = np.array([x[0] for x in candidate_embs])
        
        # Calculate cosine similarity
        sim = cosine_similarity(target_emb.reshape(1, -1), candidate_embs)[0]
        
        # Get truth labels
        labels = truth_data.iloc[idx, 1:].to_numpy()
        if labels.sum() == 0:
            continue
        
        # Rank labels by similarity
        ranked_label = labels[np.argsort(sim)[::-1]]
        ranked_label_list.append(ranked_label)
    
    # Convert to NumPy array
    ranked_label_list = np.array(ranked_label_list)

    # Calculate metrics
    return_dict = {}
    for k in [1, 2, 5]:
        return_dict[f'prec@{k}'] = precision(ranked_label_list, k)
        return_dict[f'rec@{k}'] = recall(ranked_label_list, k)
    return_dict[f'ndcg@{k}'] = ndcg(ranked_label_list, k)
    return_dict[f'MAP'] = MAP(ranked_label_list)
    
    return return_dict

np.random.seed(1)

def bootstrap_evaluate(rank_data, truth_data, test_df, n=100):
    results = []
    for i in range(n):
        #randomly sample 10 interger numbers from 0 to the length of the rank_data
        idx = np.random.choice(range(len(rank_data)), 50)
        sample_rank_data = rank_data.iloc[idx,:]
        sample_truth_data = truth_data.iloc[idx,:]
        #reset the index
        sample_rank_data = sample_rank_data.reset_index(drop=True)
        sample_truth_data = sample_truth_data.reset_index(drop=True)
        results.append(evaluate(sample_rank_data, sample_truth_data, test_df))
    results = pd.DataFrame(results)
    return results.describe().loc[['mean', 'std']]

results = bootstrap_evaluate(test_trial_list_df_rank, test_trial_list_df_truth, test_df)

#2 digits after the decimal point for every value in the DataFrame
results = results.round(3)
print(results)


