---
title: 'Transformers Models and Tokenizers'
date: 2025-04-05

parent: NLP with Hugging Face

nav_order: 3

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Transformers Models and Tokenizers
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Models

### Creating a Transformer

Initializing the BERT model is loading a configuration object.

```python
from transformers import BertConfig, BertModel
```
```python
config = BertConfig()
print(config)
```

```
BertConfig {
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": null,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.42.4",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 30522
}
```

The configuration contains many attributes, as we can see from the above output.

```python
# building model from configuration
model = BertModel(config)
```

### Model loading methods

Building model from default configuration initalizes it with random values:

```python
config = BertConfig()
model = BertModel(config)
```

Here, the model is randomly initialized. We can load the model using the `from_pretrained()` method in Transformers.


```python
model = BertModel.from_pretrained('bert-base-cased')
```

In the above code, we didn't use the `BertConfig` class; instead, we loaded the model using the `bert-base-cased` identifier. This model is now initialized with all the weights of the checkpoint.

### Saving the model

we can use `save_pretrained()` method


```python
model.save_pretrained('model_weights')
```

The above code saves teo files in a specified directory:


```python
ls model_weights
```

```
config.json  model.safetensors
```


* **Config.json**: contains metadata, like where the checkpoint originated and what Transformers version you were using when you last saved the checkpoint.

* **model.safetensors**: it contains the model's weights.


```python
cat model_weights/config.json
```
```
{
    "_name_or_path": "bert-base-cased",
    "architectures": [
    "BertModel"
    ],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": null,
    "gradient_checkpointing": false,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "torch_dtype": "float32",
    "transformers_version": "4.42.4",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 28996
}
```

### Using Transformer model for Inference

Foe example, we have a couple of sequences

```python
sequences = ["Hello", "Well done", "thank you!"]
```

Now, the tokenizer converts these sequences to vovabulary indices (i.e. input IDs).


```python
from transformers import AutoTokenizer
```

```python
checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

```python
encoded_sequences = tokenizer(
                              sequences,
                              padding=True,
                              truncation=True
                            )
```

```python
input_ids = encoded_sequences["input_ids"]
input_ids
```

```
[[101, 7592, 102, 0, 0],
    [101, 2092, 2589, 102, 0],
    [101, 4067, 2017, 999, 102]]
```

The `input_ids` is a list of encoded sequences. We will now convert it into a tensor.


```python
import torch

inputs = torch.tensor(input_ids)
```

### Using tensors as inputs to the model


```python
output = model(inputs)
print(output.last_hidden_state.shape)
```

```
torch.Size([3, 5, 768])
```

As we can see from the above output that we have,

* 3 sequences
* Length of sequence is 5, and
* Hidden size is 768.

## Tokenizers

### Word-based


```python
text = "It's his favorite sport!"

# splitting text on spaces
tokenized_text = text.split()
print(tokenized_text)
```

```
["It's", 'his', 'favorite', 'sport!']
```

Each word gets assigned an ID, starting from zero and going up to the size of the vocabulary. The model utilizes these IDs to recognize each word.

### Character-based

Split text into characters instead of words. It has two major benefits:

* Now, vocabulary is much smaller.
* There are fewer out-of-vocab tokens, as every word can be created from characters.

**Limitation**:

* It will end up with a very large amount of tokens to be processed.
* It's less meaningful as each character doesn't mean a lot on its own.

### Subword Tokenization

On a principle that frequently used words should not be split into smaller subwords. However, rare words should be split into meaningful words.

### Loading and Saving Tokenization

Loading the BERT tokenizer using `BertTokenizer` class


```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
```

We can also load tokenizer using `AutoTokenizer` class


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
```

Now, we will be using the tokenizer defined above on the sequence


```python
tokenizer('Today, I learned techniques for natural language processing.')
```

```
{'input_ids': [101, 3570, 117, 146, 3560, 4884, 1111, 2379, 1846, 6165, 119, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Saving a tokenizer

```python
tokenizer.save_pretrained('tokenizer-dir')
```

```
('tokenizer-dir/tokenizer_config.json',
    'tokenizer-dir/special_tokens_map.json',
    'tokenizer-dir/vocab.txt',
    'tokenizer-dir/added_tokens.json',
    'tokenizer-dir/tokenizer.json')
```

## Encoding

Encoding is the process of translating text into numbers. It is a two-step process:

*   Tokenization, and
*   Conversion to input IDs



### Tokenization

We will use `tokenize()` method for converting sequence into tokens


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

sequence =  "It's his favorite sport!"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

```
['It', "'", 's', 'his', 'favorite', 'sport', '!']
```

### Converting Tokens into input IDs

Here, we will use `convert_tokens_to_ids()` method


```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
```

```
[1135, 112, 188, 1117, 5095, 4799, 106]
```

## Decoding

Decoding is a process of converting ids back to the string. Let's do it using `decode()` method


```python
decoded_sequence = tokenizer.decode(input_ids)
print(decoded_sequence)
```

```
It's his favorite sport!
```

{: .note }
This decoder not only converts the input ids back to the sequence but also puts together the tokens that were part of the same words.
