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

## Classifiers

We normalized the annotation schemes of the datasets used in our experiments to be *toxic* vs *non-toxic*. For instance, when we used our <a href="https://github.com/HKUST-KnowComp/MLMA_hate_speech"> MLMA </a> dataset, we used the taxonomy explained in the discription (*non-toxic* for single-labeled normal tweets, and *toxic* for the other tweets).

## Citation
    @inproceedings{ousidhoum-etal-2021-probing,
    title = "Probing Toxic Content in Large Pre-Trained Language Models",
    author = "Ousidhoum, Nedjma  and
      Zhao, Xinran  and
      Fang, Tianqing  and
      Song, Yangqiu  and
      Yeung, Dit-Yan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.329",
    doi = "10.18653/v1/2021.acl-long.329",
    pages = "4262--4274",
    abstract = "Large pre-trained language models (PTLMs) have been shown to carry biases towards different social groups which leads to the reproduction of stereotypical and toxic content by major NLP systems. We propose a method based on logistic regression classifiers to probe English, French, and Arabic PTLMs and quantify the potentially harmful content that they convey with respect to a set of templates. The templates are prompted by a name of a social group followed by a cause-effect relation. We use PTLMs to predict masked tokens at the end of a sentence in order to examine how likely they enable toxicity towards specific communities. We shed the light on how such negative content can be triggered within unrelated and benign contexts based on evidence from a large-scale study, then we explain how to take advantage of our methodology to assess and mitigate the toxicity transmitted by PTLMs.",
}

