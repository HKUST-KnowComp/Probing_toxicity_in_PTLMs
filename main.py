""" Finetuning BERT/RoBERTa models on WinoGrande. """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import random
import math
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

MODELS = {
    'roberta': 'roberta-large',
    'bert': 'bert-large-uncased',
    'gpt2': "gpt2-large"
    # gpt2, bert,roberta: French, Arabic
}

MASK = '[MASK]'

import transformers
from transformers import pipeline 
from transformers import (BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM) # , BertForMultipleChoice, RobertForMultipleChoice
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import CamembertForMaskedLM, CamembertTokenizer 

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def preds_clean(predictions):
    new_preds = []
    for x in predictions:
        if x.startswith("Ġ"):
            x = x[1:]
        new_preds.append(x)
    return new_preds


# def get_CamemBERT_predictions(sentence, camembert_fill_mask, device, k=10):
def get_camem_predictions(sent, tokenizer, model, device, k=10):

    masked_preds = []
    masked_probs = []

    sent = sent.replace('[MASK]',"<mask>")
    assert sent.count("<mask>") == 1

    input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0).to(device) # Batch size 1
    logits = model(input_ids)[0]  # The last hidden-state is the first element of the output tuple
    masked_index = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
    logits = logits[0, masked_index, :]
    prob = logits.softmax(dim=0)
    values, indices = prob.topk(k=k, dim=0)
    topk_predicted_token_bpe = " ".join(
        [tokenizer.convert_ids_to_tokens(indices[i].item()) for i in range(len(indices))]
    )
    masked_token = tokenizer.mask_token
    topk_filled_outputs = []
    for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(" ")):
        predicted_token = predicted_token_bpe.replace("\u2581", " ")
        masked_preds.append(predicted_token)
        masked_probs.append(values[index].item())

    return masked_preds, masked_probs

def get_GPT2_predictions(sentence, lm_tokenizer, lm_model, device, k=10):
    
    masked_preds = []
    masked_probs = []

    vals = sentence.split('[MASK]')
    sentence = vals[0].strip() # + lm_tokenizer.unk_token
    # sentence = sentence.replace("[MASK]", lm_tokenizer.unk_token)

    inputs = lm_tokenizer.encode(sentence)
    inputs = torch.LongTensor(inputs).unsqueeze(0).to(device)
    res = lm_model(inputs)
    temp = res[0]

    res = res[0][0][-1]
    res = torch.nn.functional.softmax(res, -1)
    probs, best_k = torch.topk(res, k)

    probs = [x.item() for x in probs]
    best_k = [int(x) for x in best_k]
    best_k = lm_tokenizer.convert_ids_to_tokens(best_k)
    best_k = preds_clean(best_k)

    masked_preds.append(best_k)
    masked_probs.append(probs)

    return masked_preds[0], masked_probs[0]


def get_MLM_predictions(sentence, lm_tokenizer, lm_model, device, k=10):
    
    masked_preds = []
    masked_probs = []

    vals = sentence.split('[MASK]')
    mask_token = lm_tokenizer.mask_token
    
    for i in range(len(vals) - 1):
        pre = f' {mask_token} '.join(vals[: i + 1]).strip()
        post = f' {mask_token} '.join(vals[i + 1:]).strip()
        target = [lm_tokenizer.mask_token]
        tokens = [lm_tokenizer.cls_token] + lm_tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += target + lm_tokenizer.tokenize(post) + [lm_tokenizer.sep_token]
        input_ids = lm_tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        res = lm_model(tens)[0][0, target_idx]
        res = torch.nn.functional.softmax(res, -1)
        probs, best_k = torch.topk(res, k)

        probs = [x.item() for x in probs]
        best_k = [int(x) for x in best_k]
        best_k = lm_tokenizer.convert_ids_to_tokens(best_k)
        best_k = preds_clean(best_k)

        masked_preds.append(best_k)
        masked_probs.append(probs)

    return masked_preds[0], masked_probs[0]


def get_choice_prediction(sentence, candidate, lm_tokenizer, lm_model):
    max_seq_length = 40
    sep_token = lm_tokenizer.sep_token  # "[SEP]"
    cls_token = lm_tokenizer.cls_token  # "[CLS]"
    pad_token = lm_tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    sequence_a_segment_id = 0
    sequence_b_segment_id = 0

    cls_token_segment_id = 0
    pad_token_segment_id = 0

    context = sentence.split("[MASK]")[0]
    tail = sentence.split("[MASK]")[1].lstrip()

    # print(context)
    op_tail = candidate.rstrip() + " " + tail
    # print(op1_tail)

    special_tokens_count = 2  # [CLS] context + option.[SEP]

    def example_to_feature(context, tail):
        context_tokens = lm_tokenizer.tokenize(context)
        tail_tokens = lm_tokenizer.tokenize(tail)

        _truncate_seq_pair(context_tokens, tail_tokens, max_seq_length - special_tokens_count)

        tokens = context_tokens
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens += tail_tokens + [sep_token]

        segment_ids += [sequence_b_segment_id] * (len(tail_tokens) + 1)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids  # id = 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)


        input_mask = [1] * len(input_ids)  # 1 for the real tokens

        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)  # id = 0

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return (tokens, input_ids, input_mask, segment_ids)

    feature = example_to_feature(context, op_tail)

    s_tokens = feature[1]
    s_mask = feature[2]
    s_seg = feature[3]

    input_ids = torch.tensor([s_tokens], dtype=torch.long).unsqueeze(0).to(device)
    masks = torch.tensor([s_mask], dtype=torch.long).unsqueeze(0).to(device)
    segs = torch.tensor([s_seg], dtype=torch.long).unsqueeze(0).to(device)
    labels = torch.tensor(1).unsqueeze(0).to(device)  # Batch size 1

    lm_model.eval()

    # print('input_ids:', input_ids)
    # print('input_masks:', masks)
    # print('segment_ids:', segs)
    # print('labels:', labels)

    with torch.no_grad():
        outputs = lm_model(input_ids=input_ids, attention_mask=masks, token_type_ids=None, labels=labels)
        loss, classification_scores = outputs[:2]

    # print(classification_scores)
    return(classification_scores.data[0][0])

def evaluate_mlm(args):
    model_proper_name = args.model_name_or_path

    if "roberta" in args.model_type:
        tokenizer = RobertaTokenizer.from_pretrained(model_proper_name, do_lower_case=True)
        # config = RobertaConfig.from_pretrained(model_proper_name, num_labels=1)
        mlm_model = RobertaForMaskedLM.from_pretrained(model_proper_name)
    
    elif "bert" in args.model_type:
        tokenizer = BertTokenizer.from_pretrained(model_proper_name)
        mlm_model = BertForMaskedLM.from_pretrained(model_proper_name)

    elif "gpt2" in args.model_type:
        tokenizer = GPT2Tokenizer.from_pretrained(model_proper_name)
        mlm_model =  GPT2LMHeadModel.from_pretrained(model_proper_name) # transformers 3.3.1

    elif "camem" in args.model_type:
        tokenizer = CamembertTokenizer.from_pretrained(model_proper_name)
        mlm_model =  CamembertForMaskedLM.from_pretrained(model_proper_name)

    return tokenizer, mlm_model


def evaluate_choice(args):
    model_proper_name = args.model_name_or_path

    if "roberta" in args.model_type:
        tokenizer = RobertaTokenizer.from_pretrained(model_proper_name, do_lower_case=True)
        # config = RobertaConfig.from_pretrained(model_proper_name, num_labels=1)
        mc_model = RobertaForMultipleChoice.from_pretrained(model_proper_name)
    
    elif "bert" in args.model_type:
        tokenizer = BertTokenizer.from_pretrained(model_proper_name)
        mc_model = BertForMultipleChoice.from_pretrained(model_proper_name)

    return tokenizer, mc_model


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="bert. roberta, gpt2, camem(french)")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut names")
    parser.add_argument("--prediction_mode", default="mlm", type=str, 
                        help="the type of prediction to make, mlm or choice")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=80, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--one_more_time", action='store_true',
                        help="discover one more token in gpt2")
    parser.add_argument('--spanstart', type=int, default=0,
                        help="if you want to choose a span of the sentences")
    parser.add_argument('--spanend', type=int, default=396000,
                        help="if you want to choose a span of the sentences")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    args.model_type = args.model_type.lower()
    # Prepare Model:
    # if args.model_type == "camem":
    #     camembert_fill_mask  = pipeline("fill-mask", model=args.model_name_or_path, tokenizer=args.model_name_or_path)

    if args.prediction_mode == "mlm":
        tokenizer, model = evaluate_mlm(args)
        model.to(device)
        model.eval()

    elif args.prediction_mode == "choice":
        tokenizer, model = evaluate_choice(args)
        model.to(device)
        model.eval()

    # Get patterns: [atomic_head]->[patterns]->[candidates]->[outputs]
    # [atomic_head] PersonX abandons ___ altogether
    # [patterns] 1,3,5 male; 2,4,6 female;
    # [patterns] 1,2 adj; 3,4 verb; 5,6 noun
    # [candidates] top 32 are gender neutral

    # sentence, sentence ouptut, data
    results = []

    sent_path = "./sentences.json"
    if args.model_type == "camem":
        sent_path = "./sentences_fr.json"
    elif "arabert" in args.model_name_or_path:
        sent_path = "./sentences_ar.json"

    with open(sent_path , "r", encoding="utf-8") as f: # would change according to the language
        sentences = json.load(f)

    # temp test
    for sent in tqdm(sentences[args.spanstart:args.spanend]):
        temp = dict() # sentence, output

        if args.model_type == "camem":
            masked_preds, masked_probs = get_camem_predictions(sent, tokenizer, model, device, k=10)
            # masked_preds, masked_probs = get_CamemBERT_predictions(sent, camembert_fill_mask, device, k=10)

        elif args.model_type == "gpt2":
            masked_preds, masked_probs = get_GPT2_predictions(sent, tokenizer, model, device, k=10)
            if args.one_more_time:
                cont_sent = sent.replace("[MASK]", masked_preds[0] + " [MASK]")
                cont_preds, cont_probs = get_GPT2_predictions(cont_sent, tokenizer, model, device, k=10)
                
                # temp["next_sentence"] = cont_sent 
                temp["next_predictions"] = cont_preds
                temp["next_scores"] = cont_probs
        else:
            masked_preds, masked_probs = get_MLM_predictions(sent, tokenizer, model, device, k=10)

        # temp["predictions"] = list(zip(masked_preds, masked_probs))
        temp["sentence"] = sent
        temp["predictions"] = masked_preds
        temp["scores"] = masked_probs
        results.append(temp)
        
    logger.info("***** Experiment finished *****")

    span = str(args.spanstart)+":"+ str(args.spanend)

    if "/" in args.model_name_or_path:
        args.model_name_or_path = args.model_name_or_path.replace("/","_")

    if args.one_more_time:
        with open("./output_results_"+args.model_name_or_path + "_twice_" + span + ".json", "w", encoding="utf-8") as f:
            if args.model_type == "camem" or "arabert" in args.model_name_or_path:
                json.dump(results, f, ensure_ascii=False)
            else:
                json.dump(results, f)

    else:
        with open("./output_results_"+args.model_name_or_path + "_" + span + ".json", "w", encoding="utf-8") as f:
            if args.model_type == "camem" or "arabert" in args.model_name_or_path:
                json.dump(results, f, ensure_ascii=False)
            else:
                json.dump(results, f)

    return results


if __name__ == "__main__":
    results = main()
