# Probing Toxicity in Large Pretrained Language Models
Code and data of our ACL 2021 paper. (You can read the paper [probing_toxic_content_acl2021.pdf])

cands.zip includes the list of social groups. 

The atomic probing csv files includes the actions and the patterns.


## Requirements

Python 3.6 and above

tqdm

torch 0.4 and above

transformers 2.2.1 and above

## Examples

### Generate tokens with Roberta or BERT
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_type roberta --model_name_or_path roberta-large --output_dir ./output/test --spanstart 0 

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_type bert --model_name_or_path bert-large-uncased --output_dir ./output/test --overwrite_output_dir --spanstart 0

### Generate 2 tokens with GPT-2

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_type gpt2 --model_name_or_path gpt2-large --output_dir ./output/test --spanstart 0 --one_more_time

### Generate tokens with CamemBERT and AraBERT

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_type camem --model_name_or_path camembert/camembert-large --output_dir ./output/test --spanstart 0 --spanend 20

CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_type bert --model_name_or_path aubmindlab/bert-base-arabert --output_dir ./output/test --overwrite_output_dir --spanstart 0


