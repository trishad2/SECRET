import argparse
import torch
import transformers
#from transformers import BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import time

start = time.time()


# Initialize the parser
parser = argparse.ArgumentParser(description="Q/A Generation from context.")

# Add arguments
parser.add_argument('-n', '--name', type=str, required=True, help="description or criteria")
parser.add_argument('-f', '--file', type=str, help="file name")

# Parse arguments
args = parser.parse_args()




#python3 q_a_generation_arg.py -n criteria -f demo_train_data.csv





#model_id = "aaditya/OpenBioLLM-Llama3-70B"
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Set the pad_token_id (use eos_token_id or add a custom token)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# Load model and wrap in DataParallel
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)

# Use the underlying model (model.module) for the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model.module if isinstance(model, torch.nn.DataParallel) else model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # Specify GPU device
)

#function to extract questions and answers from the text using only the medical entities

def questions_answers(text):
    messages = [
    #    {"role": "system", "content": "You are an expert in creating key questions from a medical text and extract the answers from the text. Extract 3-10 Q/A pairs without repititions of key entities in the Q/As. Avoid general questions like 'What is the exclusion criteria?'. Make sure an answer is NO MORE than 5 tokens/words. Output as json format like this: {'Question': 'question1', 'Answer': 'answer1', 'Question': 'question2' , 'Answer': 'answer2', ...} \n Input: "},
        {"role": "system", "content": "You are an expert in creating key questions from a medical text and extract the answers from the text. Extract 3-10 Q/A pairs without repititions of key entities in the Q/As. Avoid general questions like 'What is the exclusion criteria?'. Make sure an answer is NO MORE than 5 tokens/words. Output ONLY json formated Q/A pairs like this: {'Question': 'question1', 'Answer': 'answer1'} \n {'Question': 'question2' , 'Answer': 'answer2'} \n ... \n Input: "},
        {"role": "user", "content": text}]
    
    prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    #print(outputs[0]["generated_text"][len(prompt):])
    return outputs[0]["generated_text"][len(prompt):]


file = args.file

#data is in ../data folder
df = pd.read_csv("../data/" + file)


tqdm.pandas()

type_of_text = args.name
df['q_a_'+ type_of_text] = df[type_of_text].progress_apply(questions_answers)

#select substring before .csv
file = file[:-4]

#save to csv
df.to_csv(file + '_' + type_of_text+ '.csv', index=False)

#print total time
print("Total time: ", time.time()-start)