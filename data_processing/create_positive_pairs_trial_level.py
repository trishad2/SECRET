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
warnings.filterwarnings("ignore")

# Initialize the parser
parser = argparse.ArgumentParser(description="create positive pairs for semisupevised learning")

# Add arguments
parser.add_argument('-u', '--unsupervised_file', type=str, help="unsupervised dataset")
parser.add_argument('-s', '--supervised_file', type=str, help="supervised dataset")
parser.add_argument('-p', '--pairs', type=str, help="positive pairs from supervised dataset")

#example command: python3 create_positive_pairs_trial_level.py -u '../data/demo_train_data.csv' -s '../data/train_data_labeled.csv' -p '../data/positive_pairs.pkl'
# Parse arguments
args = parser.parse_args()

#read the training dataset
file = args.supervised_file
df = pd.read_csv(file)

file = args.unsupervised_file
df_unsupervised = pd.read_csv(file)


print(df.columns)


with open(args.pairs, 'rb') as f:
    positive_pairs = pickle.load(f)

print(len(positive_pairs))
#functions ---------------------------------------------------------------- start


def create_single_list(row):
    single_list = []
    for q_a_pairs in [row['q_a_criteria'], row['q_a_intervention'], row['q_a_disease'],  row['q_a_keywords'], row['q_a_title'], row['q_a_outcome']]:
        for pair in q_a_pairs:
            single_list.append(pair)
    return single_list

def create_positive_trial_unsupervised(row):
    # Ensure we make a deep copy of the q_a list to avoid modifying the original data
    positive_trial = row['q_a'][:]
    if positive_trial:  # Ensure the list is not empty
        #positive_trial.pop(np.random.randint(len(positive_trial)))
        positive_trial.pop(np.random.randint(min(5, len(positive_trial))))

    return positive_trial 

def create_negative_trial_unsupervised(row, df):
    # Find rows in df with the same disease
    same_disease = df[df['disease'] == row['disease']]
    # If same disease is empty, randomly select a row from df
    if same_disease.empty:
        negative_trial = df['q_a'].iloc[np.random.randint(len(df))]
    else:
        negative_trial = same_disease['q_a'].iloc[np.random.randint(len(same_disease))]
    # Return q_a of the negative trial
    return negative_trial  

"""def create_negative_trial_unsupervised(row, df):
    #randomly select a row from df where disease is not the same
    negative_trial = df['q_a'].iloc[np.random.randint(len(df))]
    return negative_trial"""

def create_positive_trial_supervised(row, positive_pairs, df):
    #positive_pairs is a list of tuples of nct_ids. We need to find the nct_id of the current row and then find the corresponding positive pair
    nct_id = row['nct_id']
    for pair in positive_pairs:
        if nct_id in pair:
            positive_trial = pair[1] if pair[0] == nct_id else pair[0]
            #return row of the positive trial
            return df[df['nct_id'] == positive_trial]['q_a'].iloc[0]
    return ''
    

#functions ---------------------------------------------------------------- end

df['q_a_criteria'] = df['q_a_criteria'].apply(ast.literal_eval)
df['q_a_criteria'] = df['q_a_criteria'].apply(lambda x: x[:10] if len(x) > 10 else x)

df['q_a_intervention'] = df['q_a_intervention'].apply(ast.literal_eval)
df['q_a_disease'] = df['q_a_disease'].apply(ast.literal_eval)
df['q_a_outcome'] = df['q_a_outcome'].apply(ast.literal_eval)
df['q_a_keywords'] = df['q_a_keywords'].apply(ast.literal_eval)
df['q_a_title'] = df['q_a_title'].apply(ast.literal_eval)
df['q_a']= df.apply(create_single_list, axis=1)

df_unsupervised['q_a_criteria'] = df_unsupervised['q_a_criteria'].apply(ast.literal_eval)
df_unsupervised['q_a_criteria'] = df_unsupervised['q_a_criteria'].apply(lambda x: x[:10] if len(x) > 10 else x)

df_unsupervised['q_a_intervention'] = df_unsupervised['q_a_intervention'].apply(ast.literal_eval)
df_unsupervised['q_a_disease'] = df_unsupervised['q_a_disease'].apply(ast.literal_eval)
df_unsupervised['q_a_outcome'] = df_unsupervised['q_a_outcome'].apply(ast.literal_eval)
df_unsupervised['q_a_keywords'] = df_unsupervised['q_a_keywords'].apply(ast.literal_eval)
df_unsupervised['q_a_title'] = df_unsupervised['q_a_title'].apply(ast.literal_eval)
df_unsupervised['q_a']= df_unsupervised.apply(create_single_list, axis=1)

print('unsupervised positive pairs........')
#unsupervised positive pairs by dropping one q/a pair
df_unsupervised['positive_trial_us'] = df_unsupervised.apply(create_positive_trial_unsupervised, axis=1)
#print length of positive_trial_us[0] vs q_a[0]
print(len(df_unsupervised['positive_trial_us'][0]), len(df_unsupervised['q_a'][0]))

print('unsupervised negative pairs........')
#negative pairs
df_unsupervised['negative_trial'] = df_unsupervised.apply(create_negative_trial_unsupervised, args=(df_unsupervised,), axis=1)

print('supervised negative pairs........')
df['negative_trial'] = df.apply(create_negative_trial_unsupervised, args=(df,), axis=1)

print('supervised positive pairs........')
#supervised positive pairs
df['positive_trial'] = df.apply(create_positive_trial_supervised, args=(positive_pairs, df,), axis=1)

trials = []
for i in range(len(df)):
    #get the q_a pairs
    q_a = df['q_a'][i]
    positive = df['positive_trial'][i]
    negative = df['negative_trial'][i]
    #create the text
    text = {"anchor": q_a, "positive": positive, "negative": negative}
    trials.append(text)
for i in range(len(df_unsupervised)):
    #get the q_a pairs
    q_a = df_unsupervised['q_a'][i]
    positive = df_unsupervised['positive_trial_us'][i]
    negative = df_unsupervised['negative_trial'][i]
    #create the text
    text = {"anchor": q_a, "positive": positive, "negative": negative}
    trials.append(text)

#save text to json
with open('../data/text_trial_level_demo.json', 'w') as f:
    json.dump(trials, f)