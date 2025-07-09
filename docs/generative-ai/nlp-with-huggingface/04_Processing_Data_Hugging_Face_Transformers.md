---
title: 'Processing Data'
date: 2025-04-05

parent: NLP with Hugging Face

nav_order: 5

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Processing Data
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>


Let's see how we can train a sequence classifier on one batch in PyTorch:


```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# create a model and tokenizer instance from pre-traiend weights
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```

```python
# define sequences
sequences = [
    "I am learning new natural language processing (NLP) techniques.",
    "This course is amazing!"
]
# create batch of these sequences
batch = tokenizer(
    sequences,
    padding=True,
    truncation=True,
    return_tensors ='pt'
)
print(batch)
```

```
{'input_ids': tensor([
    [  101,  1045,  2572,  4083,  2047,  3019,  2653,  6364,  1006, 17953,   2361,  1007,  5461,  1012,   102],
    [  101,  2023,  2607,  2003,  6429,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]]), 
    'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```


Now we define labels for each sequence.


```python
batch['labels'] = torch.tensor([1, 1])
print(batch)
```


```
{'input_ids': tensor([
    [  101,  1045,  2572,  4083,  2047,  3019,  2653,  6364,  1006, 17953,
            2361,  1007,  5461,  1012,   102],
    [  101,  2023,  2607,  2003,  6429,   999,   102,     0,     0,     0,
                0,     0,     0,     0,     0]]), 
'token_type_ids': tensor(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'labels': tensor([1, 1])}
```


It's time to define the optimizer. Here, we use `AdamW` optimizer. It is a varient of the `Adam` optimizer. You can find more details
[here](https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw).


```python
# define optimizer
optimizer = AdamW(model.parameters())

# calculate the loss
loss = model(**batch).loss

# backward propagation
loss.backward()

optimizer.step()
```


We have just seen how we can train a model, calculate the loss, and do the backward propagation. But it has just seizures, and we know our model will not give good results on that.

Now, we need to define a larger dataset and train a model on that.

## Loading dataset from Hub

So for that purpose we are using the [MRPC (Microsoft Research Paraphrse Corpus) dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52398).

* It contains 5,801 pairs of sentences.
* and a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).

This dataset was introduced in a [Automatically Constructing a Corpus of Sentential Paraphrases](https://aclanthology.org/I05-5002.pdf) paper by William Dolan and Chris Brockett.


```python
from datasets import load_dataset

raw_dataset = load_dataset('glue', 'mrpc')
print(raw_dataset)
```


```
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

We can access the sentences in our `raw_dataset` with the help of indexing, like with a dictionary.


```python
raw_train_dataset = raw_dataset["train"]
print(raw_train_dataset[0])
```


```
{
    'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
    'label': 1,
    'idx': 0
}
```

Let's explore the `features` of our training dataset to see which integer corresponds to which label.

```python
print(raw_train_dataset.features)
```



```
{
    'sentence1': Value(dtype='string', id=None),
    'sentence2': Value(dtype='string', id=None),
    'label': ClassLabel(names=['not_equivalent', 'equivalent'], id=None),
    'idx': Value(dtype='int32', id=None)
}
```


Here, `label` is of type `ClassLabel`, and `0` corresponds to `not_equivalent`, and `1` corresponds to `equivalent` here.

## Preprocessing of a Dataset

First, we tokenize the sentences with the help of `Auotokenizer` to convert text into numbers.


```python
from transformers import AutoTokenizer

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_sent1 = tokenizer(raw_dataset['train']['sentence1'])
tokenized_sent2 = tokenizer(raw_dataset['train']['sentence2'])
```

We can't just pass two sequences to the model and get a prediction of whether two sentences are paragraphs or not.

Let's see how we can handle two sequences with the below example:


```python
inputs = tokenizer("This is first sentence", "This is second sentence.")
print(inputs)
```

```
{
    'input_ids': [101, 2023, 2003, 2034, 6251, 102, 2023, 2003, 2117, 6251, 1012, 102], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```


Here, if you have noticed, `token_type_ids` actually represents which part is the first sentence and which part is the second sentence.

Now, let's decode it back to words:


```python
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
```

```
['[CLS]',
'this',
'is',
'first',
'sentence',
'[SEP]',
'this',
'is',
'second',
'sentence',
'.',
'[SEP]']
```


From the above output, we can tell that model expects sequences to be in the form of `[CLS] sentence1 [SEP] sentence2 [SEP]` for two sentences.

Now that we have observed how our tokenizer can deal with pairs of sentences, we can utilize this method to tokenize our dataset.


```python
tokenized_dataset = tokenizer(
    raw_dataset["train"]["sentence1"],
    raw_dataset["train"]["sentence2"],
    padding=True,
    truncation=True
)
```

```python
def tokenized_function(example):
  return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```

Note that here we are not adding a `padding` argument to the function as it is not efficient. We will add it later when we create a bath  and apply padding in the respective batch. So, we need to pad to the maximum length in the batch, not to the maximum length in the entire dataset. In doing so, we can save a lot of time.


```python
tokenized_dataset = raw_dataset.map(tokenized_function, batched=True)
print(tokenized_dataset)
```

```
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})
```

## Dynamic Padding

When preparing batches of data for training, we need a function to put the samples together, called a `collate function`. By default, this function converts samples to PyTorch tensors and combines them. However, if your input data varies in size, this default approach won't work well.

To handle varying sizes, you can delay adding padding until you create each batch. This minimizes unnecessary padding, speeding up training. But keep in mind, if you're using a TPU, this might cause issues since TPUs prefer consistent input shapes, even if it means adding more padding.

The 🤗 Transformers library provides a tool called `DataCollatorWithPadding` to help with this. It automatically applies the right amount of padding for each batch based on the tokenizer you use, ensuring the inputs are correctly padded where needed.


```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

To prepare some samples from our training set for batching, we first remove unnecessary columns (like `idx`, `sentence1`, and `sentence2`) because they contain strings that can't be converted into tensors. After that, we check the lengths of each entry in the batch to ensure they can be processed together.


```python
samples = tokenized_dataset["train"][:6]
samples = {k:v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])
```
```
[50, 59, 47, 67, 59, 50]
```

Dynamic padding adjusts the length of samples in a batch so that all samples are the same length, based on the longest one in that batch. In above example, the samples vary in length from 47 to 67. With dynamic padding, each sample in the batch is padded to a length of 67, the longest sample in that batch. This avoids padding all samples in the entire dataset to the length of the longest possible sample, which would waste resources.


```python
batch = data_collator(samples)
print({k:v.shape for k, v in batch.items()})
```

```
{
    'input_ids': torch.Size([6, 67]), 
    'token_type_ids': torch.Size([6, 67]), 
    'attention_mask': torch.Size([6, 67]), 
    'labels': torch.Size([6])
}
```

Well done! In this notebook we converted raw text to batches our model can deal with. We are ready to fine-tune it.
