---
title: 'Training on GPU and Accelerators'
date: 2025-04-05

parent: NLP with Hugging Face

nav_order: 7

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Complete Training with Hugging Face Transformers on GPU and Accelerators
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>


Training machine learning models, especially deep learning models, can be computationally intensive. Utilizing GPUs (Graphics Processing Units) and accelerators (like TPUs or custom hardware) significantly speeds up this process. Hugging Face Transformers provides robust support for training models on such hardware.



## Data Processing

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
```

```python
# Load the MRPC dataset from the GLUE benchmark using the Hugging Face datasets library
raw_dataset = load_dataset("glue", "mrpc")
# Define the checkpoint for the BERT model (BERT-base, uncased version) to be used in the pipeline
checkpoint = "bert-base-uncased"
```


```python
# Initialize the tokenizer using a pre-trained model checkpoint.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

```python
def tokenize_function(sample):
  """Tokenizes a pair of sentences using the tokenizer."""
  return tokenizer(sample["sentence1"], sample["sentence2"], truncation=True)
```

```python
# Tokenize the raw dataset using the tokenize_function and apply the tokenization in batches
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
# Create a data collator that will dynamically pad the tokenized sequences to the maximum length
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Postprocessing

* Remove columns that are not needed by the model (e.g., `sentence1` and `sentence2`)

* Rename the column `label` to `labels` to match the expected format for the model

* Set the format of the dataset to return PyTorch tensors


```python
tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
print(tokenized_dataset["train"].column_names)
```

```
['labels', 'input_ids', 'token_type_ids', 'attention_mask']
```


## Define Dataloaders


```python
from torch.utils.data import DataLoader

# Create a DataLoader for the training dataset with shuffling enabled
train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle = True,
    batch_size = 8,
    collate_fn = data_collator)

# Create a DataLoader for the validation dataset
valid_dataloader = DataLoader(
    tokenized_dataset["validation"],
    batch_size = 8,
    collate_fn = data_collator)
```

Inspecting a batch:


```python
for batch in train_dataloader:
  break

{k:v.shape for k, v in batch.items()}
```

```
{
  'labels': torch.Size([8]),
  'input_ids': torch.Size([8, 66]),
  'token_type_ids': torch.Size([8, 66]),
  'attention_mask': torch.Size([8, 66])
}
```


## Model Instatiation


```python
from transformers import AutoModelForSequenceClassification
# Load a pre-trained sequence classification model from the specified checkpoint,
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

```python
output = model(**batch)
print(output.loss, output.logits.shape)
```
```
tensor(1.2856, grad_fn=<NllLossBackward0>) torch.Size([8, 2])
```

## Optimizer


```python
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)
```

## Scheduler


```python
from transformers import get_scheduler

num_epochs = 5
training_steps = num_epochs * len(train_dataloader)

scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = training_steps)

print(training_steps)
```
```
2295
```


## Training Loop


```python
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
```

```
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

```python
from tqdm.auto import tqdm
progress_bar = tqdm(range(training_steps))
model.train()

for epoch in range(num_epochs):
  for batch in train_dataloader:
    batch = {k:v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
```


## Evaluation Loop


```python
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()

for batch in valid_dataloader:
    batch = {k:v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
```

```
{'accuracy': 0.8259803921568627, 'f1': 0.8826446280991737}
```


## Training Loop with Accelerate


```python
from accelerate import Accelerator
accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

```python
optimizer = AdamW(model.parameters(), lr=1e-5)
```

```python
train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(train_dataloader, valid_dataloader, model, optimizer)
```


```python
num_epochs = 5
training_steps = num_epochs * len(train_dataloader)

scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = training_steps)
```


```python
progress_bar = tqdm(range(training_steps))
model.train()

for epoch in range(num_epochs):
  for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)
```


