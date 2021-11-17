
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
from transformers import GPT2LMHeadModel, GPT2Config, GPTNeoForCausalLM
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


def tokenize_seq(sent, tokenizer, max_length):
    return tokenizer(sent, truncation=True, max_length=max_length, padding="max_length")
class ParsingDataset(Dataset):

    def __init__(self, sentences, tokenizer, tokenizer_type="bert", max_length=1000):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []


        for sentence in sentences:
            if tokenizer_type=="bert":
                encodings = tokenize_seq(sentence, tokenizer, max_length)
            else:
                encodings = tokenize_seq("<s> " + sentence + " </s>", tokenizer, max_length)

            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

import gc
gc.collect()
#evaluate function


    # Training settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#train epoch for now

def process_data(filename, tokenizer_type):
    new_token_list = set()
    with open(filename) as file:
        lines = file.readlines()
        all_sentences = [line.strip() for line in lines]
        new_sentences = []
        for each_sent in all_sentences:
            words = each_sent.split()[:500]
            new_tokens = {word for word in words if ("(_" in word or ")_" in word)}
            # print(new_tokens)
            new_token_list.update(new_tokens)

            if tokenizer_type!="tokenizer/tokenizer_bert":
                new_sent = ""
                for word in words:
                    if ("(_" not in word and ")_" not in word):
                        new_sent += word.replace("_"," ") + " "
                    else:
                        new_sent += word + " "
                new_sentences.append(new_sent.strip())
    data = all_sentences if tokenizer_type == "tokenizer/tokenizer_bert" else new_sentences

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
    parser.add_argument('--lr', type=float, metavar='LR', default=0.0005,
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
    args = parser.parse_args()

    if args.train:
        log_file = open("saved_model/" + "logging_" + args.save_model + "_" + args.model_name + '.pt', 'w')
        log_file.write("hello world!")
    else:
        log_file = open("saved_model/" + "logging_continued_" + args.save_model + "_" + args.model_name + '.pt', 'w')

    log_file.flush()
    def eval_keywords(keywords):
        model.eval()
        for keyword in keywords:
            input_seq = keyword if args.tokenizer=="tokenizer/tokenizer_bert" else "<s> " + keyword
            generated = torch.tensor(tokenizer.encode(input_seq)).unsqueeze(0)
            generated = generated.to(device)
            sample_outputs = model.generate(
                generated,
                do_sample=True,
                top_k=30,
                max_length=600,
                top_p=0.90,
                num_return_sequences=2
            )
            for i, sample_output in enumerate(sample_outputs):
                print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                log_file.write("{}: {} \n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


    keywords = ["(_ROOT (_S (_NP", "(_ROOT (_S (_S", "(_ROOT (_S (_VP", "(_ROOT", "(_ROOT"]

    # train one batch at a time
    def process_one_batch(batch):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        outputs = model(b_input_ids, attention_mask=b_masks, labels=b_labels)
        return outputs

    import time
    # do one epoch for training
    def train_epoch():
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            outputs = process_one_batch(batch)
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()

            if step%10000==9999:
                avg_train_loss = total_train_loss / 10000
                print("avg_train_loss", avg_train_loss)
                log_file.write("avg_train_loss" + str(avg_train_loss) + "\n")

                elapsed_time = format_time(time.time() - t0)
                print("elapsed time for 10k step : ", elapsed_time)
                log_file.write("elapsed time for 10k step : " + str(elapsed_time) + "\n")

                t0 = time.time()
                total_train_loss = 0

                eval_epoch()
                eval_keywords(keywords)
                model.train()
            if step%500==0:
                print("Currently at step ", step, "/", len(train_dataloader))
                log_file.write("Currently at step " + str(step) + "/" + str(len(train_dataloader))+ "\n")

        """
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("avg_train_loss", avg_train_loss)
        log_file.write("avg_train_loss", avg_train_loss)

        elapsed_time = format_time(time.time() - t0)
        print("elapsed time for 1 training epoch : ", elapsed_time)
        log_file.write("elapsed time for 1 training epoch : ", elapsed_time)
        """

    # do one epoch for eval
    def eval_epoch():
        t0 = time.time()
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            with torch.no_grad():
                outputs = process_one_batch(batch)
                loss = outputs[0]
                batch_loss = loss.item()
                total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print("avg_val_loss", avg_val_loss)
        log_file.write("avg_val_loss" + str(avg_val_loss)+ "\n")

        elapsed_time = format_time(time.time() - t0)
        print("elapsed time for 1 eval epoch : ", elapsed_time)
        log_file.write("elapsed time for 1 eval epoch : " + str(elapsed_time)+ "\n")


    tok_type = "bert" if args.tokenizer == "tokenizer/tokenizer_bert" else "difff"

    if not args.create_tokenizer:

        if args.model_name=="gpt-neo-vi-small":
            tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        val_file = args.data_gold + "vi_vlsp21_dev.brackets"


        _, val_sents = process_data(val_file, args.tokenizer)

        max_len_val = max([len(tokenizer.encode(s)) for s in val_sents])

        print(f"max_len_val {max_len_val}")
        log_file.write(f"max_len_val {max_len_val} \n")

        tok_type = "bert" if args.tokenizer == "tokenizer/tokenizer_bert" else "difff"
        val_set = ParsingDataset(val_sents, tokenizer, tokenizer_type=tok_type, max_length=max_len_val)
        validation_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=args.batch_size)

        configuration = GPT2Config(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        if args.model_name=="gpt2":
            model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        elif args.model_name=="gpt-neo-vi-small":
            model = GPTNeoForCausalLM.from_pretrained("NlpHUST/gpt-neo-vi-small")
        elif args.model_name=="gpt2-viwiki":
            model = GPT2LMHeadModel.from_pretrained('danghuy1999/gpt2-viwiki')

        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        # model.resize_token_embeddings(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)

        if args.eval:
            model.load_state_dict(torch.load(args.save_model+ "_" + args.model_name))
            """
            TODO:
            """
        if args.continue_train:
            print("Continue training...")
            log_file.write(f"max_len_val {max_len_val} \n")

            model.load_state_dict(torch.load("saved_model/"+ args.save_model+ "_" + args.model_name + '.pt'))
            eval_keywords(keywords)

    train_file_gold = args.data_gold + "vi_vlsp21_train.brackets"

    train_file_silver = args.data_silver + "vi_silver_250k.lm"

    #new_token_list, train_sents_gold = process_data(train_file_gold, args.tokenizer)
    train_sents_gold = []
    new_token_list, train_sents_silver = process_data(train_file_silver, args.tokenizer)

    train_sents = train_sents_gold + train_sents_silver

    if args.create_tokenizer:
        # add new tokens into the tokenizer
        if args.model_name=="gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        elif args.model_name=="gpt-neo-vi-small":
            tokenizer = GPT2Tokenizer.from_pretrained("NlpHUST/gpt-neo-vi-small")
        #elif args.model_name=="gpt2-viwiki":
        #    model = GPT2LMHeadModel.from_pretrained('danghuy1999/gpt2-viwiki')
        #tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        num_added_toks = tokenizer.add_tokens(list(new_token_list))
        tokenizer.add_special_tokens({
            "eos_token":"</s>",
            "bos_token":"<s>",
            "unk_token":"<unk>",
            "pad_token": "<pad>",
            "mask_token":"<mask>"
        })

        tokenizer.save_pretrained("/tokenizer_" + args.model_name+ "/")

    # sanity_check
    """
    tokens = tokenizer.encode("(_ROOT (_S (_NP (_N Số_phận )_N (_PP (_Pre của )_Pre (_NP (_Det những )_Det ")
    print(tokens)
    print(tokenizer.convert_ids_to_tokens(tokens))
    """

    #max_len_train = max([len(tokenizer.encode(s)) for s in train_sents])
    max_len_train = 500 if tok_type=="bert" else 750

    print(f"max_len_train {max_len_train}")
    log_file.write(f"max_len_val {max_len_val} \n")

    train_set = ParsingDataset(train_sents, tokenizer,tokenizer_type=tok_type, max_length=max_len_train)

    print("train_size :", len(train_sents))
    log_file.write("train_size :" + str(len(train_sents))+ "\n")

    print("val_size   :", len(val_sents))
    log_file.write("train_size :" + str(len(train_sents))+ "\n")

    gc.collect()

    train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=args.batch_size)
    print(train_set[0])

    a, b = train_set[0]
    print(tokenizer.convert_ids_to_tokens(a))

    # Create default config
    # Load pretrained gpt2
    # Create device
    # model.cuda()

    if args.train or args.continue_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print("Training epoch ", epoch, "...")
            log_file.write("Training epoch " + str(epoch) + "..." + "\n")
            train_epoch()
            eval_epoch()
            eval_keywords(keywords)
        if args.train:
            torch.save(model.state_dict(), "saved_model/" + args.save_model  + "_" + args.model_name + '.pt')
        else:
            torch.save(model.state_dict(), "saved_model/" + "continued_" + args.save_model  + "_" + args.model_name + '.pt')

    log_file.close()
if __name__ == "__main__":
    main()


