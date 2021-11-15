
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
from transformers import GPT2LMHeadModel, GPT2Config
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
"""
tokenizer.add_special_tokens({
  "eos_token": "<EOS>",
  "bos_token": "<BOS>",
  "unk_token": "<unk>",
  "pad_token": "<PAD>",
  "mask_token": "<mask>"
})
"""
filename = "stanza_dataset/vi_vlsp21_train.brackets"
new_token_list = set()
with open(filename) as file:
    lines = file.readlines()
    all_sentences = [line.strip() for line in lines]
    for each_sent in all_sentences:
        words = each_sent.split()
        new_tokens = {word for word in words if ("(_" in word or ")_" in word)}
        #print(new_tokens)
        new_token_list.update(new_tokens)


#add new tokens into the tokenizer
print(new_token_list)
num_added_toks = tokenizer.add_tokens(list(new_token_list))
#load_model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
#model.resize_token_embeddings(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))

#sanity_check
tokens = tokenizer.encode("(_ROOT (_S (_NP (_N Số_phận )_N (_PP (_Pre của )_Pre (_NP (_Det những )_Det ")
print(tokens)
print(tokenizer.convert_ids_to_tokens(tokens))
max_len = max([len(tokenizer.encode(s)) for s in all_sentences])
print(f"max_len {max_len}")


def tokenize_seq(sent, tokenizer, max_length):
    return tokenizer(sent, truncation=True, max_length=max_length, padding="max_length")
class ParsingDataset(Dataset):

    def __init__(self, sentences, tokenizer, gpt2_type="gpt2", max_length=max_len):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for sentence in sentences:
            encodings = tokenize_seq(sentence, tokenizer, max_length)

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

dataset = ParsingDataset(all_sentences, tokenizer, max_length=max_len)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])
print("train_size :",train_size)
print("val_size   :",val_size)
gc.collect()
print(dataset[0])
a,b = dataset[0]
train_dataloader = DataLoader(train_set,  sampler = RandomSampler(train_set), batch_size = 16)
validation_dataloader = DataLoader(val_set, sampler = SequentialSampler(val_set), batch_size = 16)
# Create default config
# Load pretrained gpt2

# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
model = model.to(device)


#evaluate function
def eval_keywords(keywords):
  model.eval()
  for keyword in keywords:
    input_seq = keyword
    generated = torch.tensor(tokenizer.encode(input_seq)).unsqueeze(0)
    generated = generated.to(device)
    sample_outputs = model.generate(
                                generated,
                                do_sample=True,
                                top_k=30,
                                max_length = 50,
                                top_p=0.90,
                                num_return_sequences=2
                                )
    for i, sample_output in enumerate(sample_outputs):
      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

keywords = ["(_ROOT (_S (_NP","(_ROOT (_S (_S","(_ROOT (_S (_VP"]

#train one batch at a time
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

    avg_train_loss = total_train_loss / len(train_dataloader)
    print("avg_train_loss", avg_train_loss)
    elapsed_time = format_time(time.time() - t0)
    print("elapsed time for 1 training epoch : ", elapsed_time)

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
    elapsed_time = format_time(time.time() - t0)
    print("elapsed time for 1 eval epoch : ", elapsed_time)

#train one epoch for now
train_epoch()
eval_epoch()
eval_keywords( keywords )
