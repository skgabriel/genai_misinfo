import csv
import numpy as np

file = [row for row in csv.reader(open("../data/exp_annot.csv"))]
header = file[0]

file = file[1:]
errors = []
opinion = []
commonsense = []
events = []
domain = []

for i in range(len(file)):
    #skip incomplete or misleading annotations 
    if len(file[i]) == 0 or "Misleading" in file[i][0]:
       continue
    errors.append(file[i][header.index("Inaccurate reasoning (Yes/No)")])
    opinion.append(file[i][header.index("Explanation states opinion (Yes/No)")])
    commonsense.append(file[i][header.index("Explanation uses commonsense reasoning (Yes/No)")])
    events.append(file[i][header.index("Explanation uses knowledge of specific events (Yes/No)")])
    domain.append(file[i][header.index("Explanation uses domain knowledge (Yes/No)")])

print("Total examples: " + str(len(errors)))
print("% incorrect: " + str(100 * np.mean([e.lower() == "yes" or e.lower() == "true" for e in errors])))
print("% stating opinion: " + str(100 * np.mean([e.lower() == "yes" or e.lower() == "true" for e in opinion])))
print("% using commonsense: " + str(100 * np.mean([e.lower() == "yes" or e.lower() == "true" for e in commonsense])))
print("% using event knowledge: " + str(100 * np.mean([e.lower() == "yes" or e.lower() == "true" for e in events])))
print("% using domain knowledge: " + str(100 * np.mean([e.lower() == "yes" or e.lower() == "true" for e in domain])))
