import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import random
import argparse

def format_output(sequence,model_type):
        if model_type == "t5":
           if "<pad> " in sequence:
              sequence = sequence.replace("<pad> ","")
           return sequence.replace("</s>","")
        else:
           return sequence.split("]")[-1].replace(" <|endoftext|>","")

parser = argparse.ArgumentParser(prog='MisinfoEval', description='Generate a reaction frame explanation for why a claim is true or false.')
parser.add_argument("--device",default="gpu")
parser.add_argument("--seed",default=10)
args = parser.parse_args()

random.seed(args.seed)

if args.device == "gpu":
   device = "cuda"
else:
   device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("petrichorRainbow/mrf-GPT")
model = AutoModelForCausalLM.from_pretrained("petrichorRainbow/mrf-GPT").to(device)
dims = ["[writer_intent]", "[effect_on_reader]", "[reader_action]","[pred_label]","[gold_label]","[spread]"]
domains = ["[climate]", "[covid]","[cancer]","[Other]"]
tokenizer.add_tokens(dims + domains)

prompt1 = "Write a short explanation for why the headline '"
prompt2 = "' is '"
prompt_end = ".' Do not mention that you are AI. The explanation must be less than 100 words."

file = [row for row in csv.reader(open("newsfeed_eval_set.csv"))]
header = file[0]
print(header)
file = file[1:]
file2 = csv.writer(open("mrf.csv","w"))
file2.writerow(header)

for f_ in tqdm.tqdm(file):
    explains = []
    for i in range(1,5,1):
        headline = f_[header.index("headline" + str(i))]
        label = f_[header.index("label" + str(i))]
        input_1 = tokenizer.encode(headline + " [writer_intent]", return_tensors="pt").to(device)
        input_2 = tokenizer.encode(headline + " [reader_action]", return_tensors="pt").to(device)
        output_1 = model.generate(input_ids=input_1,num_beams=5,max_length=50)
        output_2 = model.generate(input_ids=input_2,num_beams=5,max_length=50)
        output_1 = format_output(tokenizer.decode(output_1[0]),"gpt")
        output_2 = format_output(tokenizer.decode(output_2[0]),"gpt")
        explain = "This headline is " + label.lower() + "."
        if label == "true":
           explain += " <br> This headline is trying to persuade readers that " + output_1 + ". " + "It is compelling readers to " + output_2 + "."
        else:
           explain += " <br> This headline is trying to manipulate readers by implying that " + output_1 + ". " + "It is compelling readers to " + output_2 + "."
        explains.append(explain)
    f_ = f_[:header.index("explanation1")] + explains + f_[header.index("explanation5")+1:]
    file2.writerow(f_)
