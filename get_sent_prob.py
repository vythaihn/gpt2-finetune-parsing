from torch.optim.lr_scheduler import ExponentialLR
from transformers import GPT2Tokenizer
#get pretrained tokenizer
from transformers import AutoTokenizer
"""
tokenizer = GPT2Tokenizer.from_pretrained('stanza_dataset/tokenized_data')
"""
from torch.utils.data import Dataset
import torch
import datetime
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Config, GPTNeoForCausalLM, GPTNeoConfig
#configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
import argparse
"""
tokenizer.add_special_tokens({
  "eos_token": "<EOS>",
  "bos_token": "<BOS>",
  "unk_token": "<unk>",
  "pad_token": "<PAD>",
  "mask_token": "<mask>"
})
"""

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

import gc
gc.collect()
#evaluate function


    # Training settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#train epoch for now

def process_data(args, filename):
    new_token_list = set()
    with open(filename) as file:
        lines = file.readlines()
        all_sentences = [line.strip() for line in lines]
        new_sentences = []
        new_sentences_bert = []

        for each_sent in all_sentences:
            if each_sent != "":
                words = each_sent.split()
                if len(words) < args.max_length:
                    new_tokens = {word for word in words if ("(_" in word or ")_" in word)}
                    # print(new_tokens)
                    new_token_list.update(new_tokens)
                    if args.tokenizer =="tokenizer/tokenizer_bert":
                        new_sentences_bert.append(each_sent)
                    else:
                        new_sent = ""
                        for word in words:
                            if ("(_" not in word and ")_" not in word):
                                new_sent += word.replace("_"," ") + " "
                            else:
                                new_sent += word + " "
                        new_sentences.append(new_sent.strip())

    data = new_sentences_bert if args.tokenizer == "tokenizer/tokenizer_bert" else new_sentences

    return new_token_list, data

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 dataset')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for training')
    parser.add_argument('--data-gold', default='stanza_dataset/gold/',
                        help='directory that contains cifar-10-batches-py/ '
                             '(downloaded automatically if necessary)')
    parser.add_argument('--data-silver', default='stanza_dataset/silver/',
                        help='directory that contains cifar-10-batches-py/ '
                             '(downloaded automatically if necessary)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, metavar='LR', default=0.00005,
                        help='learning rate')
    parser.add_argument('--save-model', default="test",
                        help='saves the current model at path')
    parser.add_argument('--continue-train', action='store_true', default=False,
                        help='saves the current model at path')
    parser.add_argument('--tokenizer', default="tokenizer/tokenizer_bert",
                        help='saves the current model at path')
    parser.add_argument('--model-name', default="gpt2", choices=["gpt2", "gpt-neo-vi-small", "gpt2-viwiki"],
                        help='saves the current model at path')
    parser.add_argument('--train', action='store_true', default=False,
                        help='saves the current model at path')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='saves the current model at path')
    parser.add_argument('--create-tokenizer', default=False, action='store_true',
                        help='saves the current model at path')
    parser.add_argument('--max-length', default=512, type=int,
                        help='saves the current model at path')
    args = parser.parse_args()

    if args.train:
        log_file = open("saved_model/" + "logging_" + args.save_model + "_" + args.model_name + '.pt', 'w')
        log_file.write("hello world!")
    else:
        log_file = open("saved_model/" + "logging_continued_" + args.save_model + "_" + args.model_name + '.pt', 'w')

    log_file.flush()

    def sent_scoring(sentence, tokenizer_type):
        assert model is not None
        assert tokenizer is not None
        if tokenizer_type == "bert":
            encodings = torch.tensor(tokenizer.encode(sentence, truncation = True, max_length = 1000)).unsqueeze(0)
        else:
            words = sentence.split()
            processed_sent = ' '.join([word.replace("_", " ") if ("(_" not in word and ")_" not in word) else word for word in words]).strip()
            encodings = torch.tensor(tokenizer.encode("<s> " + processed_sent + " </s>", truncation = True, max_length = 1000)).unsqueeze(0)

        error = [0 for v in encodings if v is None]

        if error == []:
            # input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
            input_ids = encodings.to('cuda')
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            sentence_prob = loss.item()
            return np.exp(sentence_prob)
        else:
            return float("inf")



    keywords = ["(_ROOT (_S (_NP", "(_ROOT (_S (_S", "(_ROOT (_S (_VP", "(_ROOT", "(_ROOT"]

    tok_type = "bert" if args.tokenizer == "tokenizer/tokenizer_bert" else "difff"

    if tok_type=="bert":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)

    configuration_GPT2 = GPT2Config(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    configuration_GPT2_neo = GPTNeoConfig.from_pretrained("NlpHUST/gpt-neo-vi-small", output_hidden_states=False)
    configuration_GPT2_neo.bos_token_id = tokenizer.bos_token_id
    configuration_GPT2_neo.eos_token_id = tokenizer.eos_token_id
    configuration_GPT2_neo.vocab_size = tokenizer.vocab_size

    if args.model_name=="gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration_GPT2)
    elif args.model_name=="gpt-neo-vi-small":
        model = GPTNeoForCausalLM.from_pretrained("NlpHUST/gpt-neo-vi-small",config=configuration_GPT2_neo)
    elif args.model_name=="gpt2-viwiki":
        model = GPT2LMHeadModel.from_pretrained('danghuy1999/gpt2-viwiki', config=configuration_GPT2)

    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    # model.resize_token_embeddings(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.eval()

    sentences = ['(_ROOT (_S (_CCONJ Nh??ng )_CCONJ (_PP (_ADP tr??n )_ADP (_NP (_NOUN m???nh )_NOUN (_NOUN ?????t )_NOUN (_NP (_NOUN ?????c_Ph??? )_NOUN )_NP (_PROPN n??y )_PROPN )_NP )_PP (_VP (_X \
v???n )_X (_X c??n )_X (_AP (_ADJ n???ng )_ADJ )_AP (_NP (_DET nh???ng )_DET (_NOUN ??au_th????ng )_NOUN )_NP )_VP (_PUNCT , )_PUNCT (_NP (_NOUN ng??y )_NOUN (_PROPN t???ng )_PROPN \
(_NOUN ng??y )_NOUN )_NP (_S (_NP (_NOUN m??u )_NOUN )_NP (_VP (_X v???n )_X (_VERB r??i )_VERB )_VP )_S (_PUNCT , )_PUNCT (_S (_NP (_NOUN x????ng )_NOUN )_NP (_VP (_X v???n )_X\
 (_VERB ????? )_VERB )_VP )_S (_PUNCT . )_PUNCT )_S )_ROOT',
                 '(_ROOT (_S (_VP (_VERB C?? )_VERB (_NP (_NOUN c??i )_NOUN (_PROPN g?? )_PROPN (_VP (_VERB mong_?????i )_VERB (_VERB tha_thi???t )_VERB )_VP )_NP (_PP (_ADP trong )_ADP (_NP (_N\
OUN l??ng )_NOUN )_NP )_PP )_VP (_PUNCT . )_PUNCT )_S )_ROOT',
                 '(_ROOT (_S (_NP (_PROPN N?? )_PROPN )_NP (_VP (_VERB t???c_gi???n )_VERB (_PP (_ADP tr?????c )_ADP (_NP (_VERB h??nh_?????ng )_VERB (_PP (_ADP c???a )_ADP (_NP (_PROPN ch??ng_t??i )_PR\
OPN )_NP )_PP )_NP )_PP )_VP (_PUNCT . )_PUNCT )_S )_ROOT',
                 '(_ROOT (_S (_NP (_PROPN N?? )_PROPN )_NP (_VP (_VERB t???c )_VERB (_VERB gi???n )_VERB (_PP (_ADP tr?????c )_ADP (_NP (_VERB h??nh_?????ng )_VERB (_PP (_ADP c???a )_ADP (_NP (_PROPN ch??ng PR\
OPN (_NOUN t )_NOUN )_NP )_PP )_NP )_PP )_VP (_PUNCT . )_PUNCT )_S )_ROOT',
                 '(_ROOT (_S (_NP (_NOUN N?? )_NOUN )_NP (_VERB l?? )_VERB (_VP (_DET con )_DET (_NOUN m??o )_NOUN )_VP )_S )_ROOT',
                 '(_ROOT (_S (_NP (_PROPN N?? )_PROPN )_NP (_AUX l?? )_AUX (_NP (_NOUN con )_NOUN (_NOUN m??o )_NOUN )_NP )_S )_ROOT'
                 ]


    model.load_state_dict(torch.load("saved_model/" +  args.save_model  + "_" + args.model_name + '.pt'))
    """
    TODO:
    """
    min_score = float("inf")
    best_sent = None
    for idx, sent in enumerate(sentences):
        score = sent_scoring(sent, tokenizer_type=tok_type)
        print("score for ", idx, " is ", score)
        if score < min_score:
            min_score = score
            best_sent = idx

    log_file.close()
if __name__ == "__main__":
    main()


