import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import openai
import tqdm
import argparse

parser = argparse.ArgumentParser(prog='MisinfoEval', description='Generate a gpt-4 explanation for why a claim is true or false.')
parser.add_argument("--api_key")
parser.add_argument("--api_type",default="open_ai")
parser.add_argument("--api_version",default="")
parser.add_argument("--gpt4_engine",default="gpt-4")
parser.add_argument("--seed",default=10)
args = parser.parse_args()

random.seed(args.seed)
openai.api_key = args.api_key
openai.api_type = args.api_type
openai.api_version = args.api_version
ENGINE_GPT4 = args.gpt4_engine

prompt1 = "Write a short explanation for why the headline '"
prompt2 = "' is '"
prompt_end = ".' Do not mention that you are AI. The explanation must be less than 100 words."

file = [row for row in csv.reader(open("../data/newsfeed_eval_set.csv"))]
header = file[0]
print(header)
file = file[1:]
file2 = csv.writer(open("gpt-4.csv","w"))
file2.writerow(header)

for f_ in tqdm.tqdm(file):
    explains = []
    for i in range(1,5,1):
        headline = f_[header.index("headline" + str(i))]
        label = f_[header.index("label" + str(i))]
        explain1 = "This headline is " + label.lower() + "."
        explain2 = openai.ChatCompletion.create(model=ENGINE_GPT4,messages=[{"role":"user","content": prompt1 + headline + prompt2 + label.lower() + prompt_end}], seed=args.seed, max_tokens=1000)["choices"][0]["message"]["content"] 
        explains.append(explain1 + " " + explain2.replace("\n"," <br> "))
    f_ = f_[:header.index("explanation1")] + explains + f_[header.index("explanation5")+1:]
    file2.writerow(f_)
