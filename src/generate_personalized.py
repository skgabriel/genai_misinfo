import csv
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import tqdm
import argparse

parser = argparse.ArgumentParser(prog='MisinfoEval', description='Generate a personalized explanation for why a claim is true or false.')
parser.add_argument("--api_key")
parser.add_argument("--api_type",default="open_ai")
parser.add_argument("--api_version",default="")
parser.add_argument("--gpt4_engine",default="gpt-4")
parser.add_argument("--group",default="conservative_white_educated_male_18-29")
parser.add_argument("--seed",default=10)
args = parser.parse_args()

openai.api_key = args.api_key
openai.api_type = args.api_type
openai.api_version = args.api_version
ENGINE_GPT4 = args.gpt4_engine

random.seed(args.seed)
group = args.group

prompt1 = "Write a short explanation for why the headline '"
prompt2 = "' is '"
prompt3 = " that will appeal to an <education>, <gender>, <race> <age> year old reader with <politics> political beliefs. Do not mention the type of reader. Do not mention that you are AI. The explanation must be less than 100 words."

params = group.split("_")
prompt_end = prompt3.replace("<education>",params[2])
prompt_end = prompt_end.replace("<politics>",params[0])
prompt_end = prompt_end.replace("<gender>",params[-2])
prompt_end = prompt_end.replace("<age>",params[-1])
prompt_end = prompt_end.replace("<race>",params[1])

file = [row for row in csv.reader(open("newsfeed_eval_set.csv"))]
header = file[0]
print(header)
file = file[1:]
file2 = csv.writer(open("personalized.csv","w"))
file2.writerow(header)

for f_ in tqdm.tqdm(file):
    explains = []
    for i in range(1,5,1):
        headline = f_[header.index("headline" + str(i))]
        label = f_[header.index("label" + str(i))]
        explain1 = "This headline is " + label.lower() + "."
        explain2 = openai.ChatCompletion.create(model=ENGINE_GPT4,messages=[{"role":"user","content": prompt1 + headline + prompt2 + label.lower() + prompt_end}], max_tokens=1000,seed=args.seed)["choices"][0]["message"]["content"] 
        explains.append(explain1 + " " + explain2.replace("\n"," <br> "))
    f_ = f_[:header.index("explanation1")] + explains + f_[header.index("explanation5")+1:]
    file2.writerow(f_)
