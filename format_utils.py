import json
import random
from tqdm import trange, tqdm

def read_datasets(atomic_path, candidate_path):

    atomic_dataset = dict()

    with open(atomic_path, "r", encoding="utf-8") as f:
        lines = f.readlines() # line 0 is header
        for i, line in enumerate(lines):
            if i == 0:
                continue

            if i == 1001:
                break

            line = line.strip()
            patterns = line.split(",")

            atomic_dataset[patterns[-1]] = patterns[:-1]
    
    cands = dict()
    cands["neutral"] = []
    cands["male"] = []
    cands["female"] = []
    
    for key in ["neutral", "male", "female"]:
        with open(candidate_path+key+".txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.lower().strip()
                cands[key].append(line)

    return atomic_dataset, cands


def generate_sentences(atomic_dataset, cands):
    # info: original_head, pattern, gender, candidates(person X, person Y, personZ), type(adj, verb, noun)
    
    sentences = []
    sentence_info = []
    signal = 0
    
    for key in tqdm(atomic_dataset.keys()):
        value = atomic_dataset[key]

        for i, pattern in enumerate(value):
            
            gender = ""
            pos_tag = ""
            this_cands = []
            
            if i%6 in [0,2,4]:
                gender = "male"
                this_cands = cands["neutral"] + cands["male"]
            elif i%6 in [1,3,5]:
                gender = "female"
                this_cands = cands["neutral"] + cands["female"]
            
            if i&6 in [0,1]:
                pos_tag = "adj"
            if i%6 in [2,3]:
                pos_tag = "verb"
            if i%6 in [4,5]:
                pos_tag = "noun"
                
            for cand in this_cands:
                temp = ""
                temp_info = dict()

                temp = pattern
                temp = temp.replace("PersonX", cand)

                temp_info["original_head"] = key
                temp_info["pattern"] = pattern
                temp_info["gender"] = gender
                temp_info["type"] = pos_tag
                temp_info["personX"] = cand

                sentences.append(temp)
                sentence_info.append(temp_info)
    
    print("Sentence Collected: ", len(sentences))
    
    return sentences, sentence_info


atomic_path = "./atomic_probing.csv"
candidate_path = "./cands/"

atomic_dataset, cands = read_datasets(atomic_path, candidate_path)
sentences, sentence_info =  generate_sentences(atomic_dataset, cands)

print(len(sentences))
print(sentences[0])

with open("./sentences.json", "w", encoding="utf-8") as f:
    json.dump(sentences, f)

with open("./sentence_info.json", "w", encoding="utf-8") as f:
    json.dump(sentence_info, f)