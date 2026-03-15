import os
import re
import json
from pathlib import Path
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TRANSFORMERS_CACHE'] = '../cache'

DATA_PATH = Path('../hate-speech-multi/data')
DATASET_FILES = [
    'lith_dataset_wide_countries.csv',
    'lith_dataset_wide_orientation.csv',
    'lith_dataset_wide_political.csv',
    'lith_dataset_wide_race.csv'
]
BATCH_SIZE = 5
SYSTEM_PROMPT = "You are an expert in hate speech detection and generation for Lithuanian language."
USER_PROMPT_1 = "Below there are instances of hate speech comments. Each of them can have one or more labels assigned to it, separated by comma. This corresponds to multilabel classification task."
USER_PROMPT_2 = """
For each comment, generate at least 5 comments which have the same semantic meaning and would have the same labels assigned to them. Make sure to follow the following guidelines:
* Use rude, foul and explicit language where applicable, as well as swearing, to make the comments sound natural. 
* Do not invent new Lithuanian words which are not present in the vocabulary.   
* The generated comments should vary in tone, severity, or lexical variation. The tone should be adjusted to to the original comment as much as possible 
* Diacritics could also be removed for some of the generated comments to look comment more natural. In that caes, diacritics should be removed for the complete comment
* Very long comments should be summarized or paraphrased into to shorter ones. 
Do not output any intermediate infomation, only final results. Return these results in JSON format, which must contain:
* index field
* original_comment field which contains the original comment
* labels field which stores the list of labels
* `generated_comments` which would contain the list of generated comments    
"""


def load_dataset(filename):
    data = pd.read_csv(filename)
    labels = set(data.columns.tolist()) - {'text'}
    data['labels'] = data.apply(lambda x: [col for col in labels if x[col] == 1], axis=1)
    items = data[['text', 'labels']].to_dict(orient='records')
    items = list(filter(lambda x: len(x['labels']) > 0, items))
    return items

def parse_output(output):
    pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = pattern.search(output)
    if match:
        json_string = match.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

def decensore(text):
    replacements = {
        'n***ui': 'nachui',
        'n**ui': 'nahui',
        'b**t': 'blet',
        'p**ti': 'pisti',
        'py*****': 'pyderai',
        'n*ger': 'niger',
        'š*d': 'šūd'
    }
    for replaced, replacement in replacements.items():
        text = re.sub(replaced.replace('*', r'\*'), replacement, text, flags=re.IGNORECASE)
    return text

def augment_dataset(filename, pipe):
    items = load_dataset(filename)
    all_results = []
    for ind in tqdm(range(0, len(items), BATCH_SIZE)):
        inputs = "".join([f"""
            Comment: {x['text']}
            Index: {i+ind} 
            Labels: {x["labels"]}
        """ for i, x in enumerate(items[ind:ind+BATCH_SIZE])
        ])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text",  "text": USER_PROMPT_1},
                {"type": "text", "text": inputs},
                {"type": "text", "text": USER_PROMPT_2}
            ]}
        ]
        output = pipe(text=messages, max_new_tokens=8192)
        output = output[0]["generated_text"][-1]["content"]
        json_output = parse_output(output)
        all_results.extend(json_output)
    output_file = os.path.basename(filename).split('.')[0]            
    with open(f"{output_file}_generated.json", "w", encoding="utf8") as f:
        json.dump(all_results, f)

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-27b-it",
    device="cuda",
    torch_dtype=torch.bfloat16
)
for dataset in DATASET_FILES:
    print("Processing dataset", dataset)
    augment_dataset(DATA_PATH/dataset, pipe)

