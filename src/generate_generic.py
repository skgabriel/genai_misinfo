import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import random
import argparse

parser = argparse.ArgumentParser(prog='MisinfoEval', description='Generate either a generic human or AI explanation for why a claim is true or false.')
parser.add_argument("--seed",default=10)
parser.add_argument("--type",default="ai")
args = parser.parse_args()

file = [row for row in csv.reader(open("../data/newsfeed_eval_set.csv"))]
header = file[0]
print(header)
file = file[1:]
file2 = csv.writer(open("generic_" + args.type + ".csv","w"))
file2.writerow(header)

for f_ in tqdm.tqdm(file):
    explains = []
    for i in range(1,5,1):
        headline = f_[header.index("headline" + str(i))]
        label = f_[header.index("label" + str(i))]
        explain = "This headline is " + label.lower() + "."
        if args.type == "human":
           source = "non-partisan fact-checkers."
        else:
           source = "an AI model trained on a large-scale corpus of web data."
        if label.lower() == "true":
           explain += " <br> This headline has been verified by " + source
        else:
           explain += " <br> This headline has been refuted by " + source  
        explains.append(explain)
    f_ = f_[:header.index("explanation1")] + explains + f_[header.index("explanation5")+1:]
    file2.writerow(f_)
