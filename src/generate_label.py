
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import random
import argparse

parser = argparse.ArgumentParser(prog='MisinfoEval', description='Show whether the claim is true or false wit>
parser.add_argument("--seed",default=10)
args = parser.parse_args()

file = [row for row in csv.reader(open("newsfeed_eval_set.csv"))]
header = file[0]
file = file[1:]
file2 = csv.writer(open("label.csv","w"))
file2.writerow(header)

for f_ in tqdm.tqdm(file):
    explains = []
    for i in range(1,5,1):
        headline = f_[header.index("headline" + str(i))]
        label = f_[header.index("label" + str(i))]
        explain = "This headline is " + label.lower() + "."
        explains.append(explain)
    f_ = f_[:header.index("explanation1")] + explains + f_[header.index("explanation5")+1:]
    file2.writerow(f_)

