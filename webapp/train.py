import subprocess
import os
import json
import torch
from pathlib import Path
from transformers import (Trainer,
                          pipeline,
                          T5Config,
                          TrainingArguments,
                          T5ForConditionalGeneration,
                          T5TokenizerFast,
                          LineByLineTextDataset,
                          DataCollatorForLanguageModeling)

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer

lang = "Python"
#lang = "Java"
#lang = "Javascript"
#lang = "Go"
subprocess.call(["wget", f"https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{lang}.zip"])
subprocess.call(["unzip", f"/content/{lang}.zip"])
log_dir = "/content/log"
data_dir = "/content/data"
model_dir = "/content/model"
tokenizer_dir = "/content/tokenizer"

def prepare_text(dir_path):
  for path in os.listdir(dir_path):
    os.system(f"gunzip -k {dir_path}/{path}")

  texts = ""
  for path in os.listdir(dir_path):
    if path.endswith(".jsonl"):
      with open(dir_path + "/" + path, 'r') as f:
        sample_file = f.readlines()
        for sample in sample_file:
          obj = json.loads(sample)
          texts += obj["original_string"].replace("\n", "").replace("\t", "") + "\n"
  return texts

train1_texts = prepare_text(f"/content/{lang}/final/jsonl/train")
train2_texts = prepare_text(f"/content/{lang}/final/jsonl/valid")
train_texts = train1_texts + "\n" + train2_texts
valid_texts = prepare_text(f"/content/{lang}/final/jsonl/test")

for path, text in zip(["train_texts.txt", "valid_texts.txt"], 
                      [train_texts, valid_texts]):
  with open(f"{data_dir}/{path}","w") as f:
    f.write(text)

paths = [str(x) for x in Path(f"{data_dir}/").glob("**/*.txt")]
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model(tokenizer_dir)

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

config = T5Config(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = T5TokenizerFast.from_pretrained(tokenizer_dir, max_len=512)

model = T5ForConditionalGeneration(config=config)
model.num_parameters()

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=f"{data_dir}/train_texts.txt",
    block_size=128,
)

test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=f"{data_dir}/valid_texts.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=model_dir,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_gpu_train_batch_size=64,
    save_steps=5000,
    do_eval=True,
    logging_dir=log_dir,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset = test_dataset
)

trainer.train()
trainer.save_model(model_dir)
tokenizer.save_pretrained(tokenizer_dir)