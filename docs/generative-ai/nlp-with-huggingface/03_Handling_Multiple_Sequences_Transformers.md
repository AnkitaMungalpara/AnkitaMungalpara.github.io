---
title: 'Handling Multiple Sequences'
date: 2025-04-05

parent: NLP with Hugging Face

nav_order: 4

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---


# Handling Multiple Sequences
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Create batch of inputs and send it to Model

First, let's convert the list of numbers into a tensor for the sample sequence and send it to the model.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```


```python
sequence = "I am currently learning machine learning."

# create tokens
tokens = tokenizer.tokenize(sequence)
print(tokens)
```

```
['i', 'am', 'currently', 'learning', 'machine', 'learning', '.']
```

```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
```

```
[1045, 2572, 2747, 4083, 3698, 4083, 1012]
```

```python
ids = torch.tensor(input_ids)
print(ids)
```

```
tensor([1045, 2572, 2747, 4083, 3698, 4083, 1012])
```

```python
model(ids)
```

```
---------------------------------------------------------------------------

Traceback (most recent call last)

<ipython-input-8-527d0145ab42> in <cell line: 1>()
----> 1 model(ids)

/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
    1530             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
    1531         else:
-> 1532             return self._call_impl(*args, **kwargs)
    1533 
    1534     def _call_impl(self, *args, **kwargs):

/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
    1539                 or _global_backward_pre_hooks or _global_backward_hooks
    1540                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1541             return forward_call(*args, **kwargs)
    1542 
    1543         try:

/usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
    988         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    989 
--> 990         distilbert_output = self.distilbert(
    991             input_ids=input_ids,
    992             attention_mask=attention_mask,

/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
    1530             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
    1531         else:
-> 1532             return self._call_impl(*args, **kwargs)
    1533 
    1534     def _call_impl(self, *args, **kwargs):

/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
    1539                 or _global_backward_pre_hooks or _global_backward_hooks
    1540                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1541             return forward_call(*args, **kwargs)
    1542 
    1543         try:

/usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    788             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    789         elif input_ids is not None:
--> 790             self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    791             input_shape = input_ids.size()
    792         elif inputs_embeds is not None:

/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py in warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask)
    4539 
    4540         # Check only the first and last input IDs to reduce overhead.
-> 4541         if self.config.pad_token_id in input_ids[:, [-1, 0]]:
    4542             warn_string = (
    4543                 "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "

IndexError: too many indices for tensor of dimension 1
```

## Why is Model failing?

Here, we sent one sequence to the model, and Transformers models require many sentences by default. Let's see how `tokenizer` works by returning `PyTorch` tensor format.


```python
tokenized_sequence = tokenizer(sequence, return_tensors = 'pt')
print(tokenized_sequence['input_ids'])
```

```
tensor([[ 101, 1045, 2572, 2747, 4083, 3698, 4083, 1012,  102]])
```

From the output we got from the tokenizer, we can clearly see that the tokenizer didn't just convert input IDs into tensors, but it added extra dimension on top of it.

Let's try again by adding a dimension to the list of input IDs.


```python
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```


```python
# create tokens
tokens = tokenizer.tokenize(sequence)
print(tokens)
```

```
['i', 'am', 'currently', 'learning', 'machine', 'learning', '.']
```


```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
```

```
[1045, 2572, 2747, 4083, 3698, 4083, 1012]
```

```python
ids = torch.tensor([input_ids])
print(ids)
```

```
tensor([[1045, 2572, 2747, 4083, 3698, 4083, 1012]])
```

```python
output = model(ids)
print(output.logits)
```

```
tensor([[-0.6310,  0.7762]], grad_fn=<AddmmBackward0>)
```

## Batching

It is the process of sending many sequences to the model at once. We have just built a batch with a single sequence.


### Padding the sequences

`padding` make sure all our sequences have the same length as we make our tensors into rectangles by adding a special token called a `padding token`.

The padding token ID can be found in `pad_token_id`.

For example, if we have a batch of sequences having input IDs as below:

```
batch_ids = [
    [100, 100, 100],
    [100, 100]
    ]
```

Now, let's send our sequences to the model individually and then in the form of batches.



```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```

```python
seq1_ids = [[100, 100, 100]]
seq2_ids = [[100, 100]]

batch_ids = [
  [100, 100, 100],
  [100, 100, tokenizer.pad_token_id]
]
```

```python
print(model(torch.tensor(seq1_ids)).logits)
print(model(torch.tensor(seq2_ids)).logits)
print(model(torch.tensor(batch_ids)).logits)
```

```
tensor([[ 1.4738, -1.3271]], grad_fn=<AddmmBackward0>)
tensor([[ 1.2205, -1.1099]], grad_fn=<AddmmBackward0>)
tensor([[ 1.4738, -1.3271],
        [ 1.7130, -1.4950]], grad_fn=<AddmmBackward0>)
```

From the above output, we observe that we got completely different results for our second sequence. It is because our Transformer models take padding tokens into consideration while they attend to all tokens of a sequence.

So, we explicitly need to tell attention layers to ignore padding tokens we applied to the sequence for getting the same results for the second sequence passed in a batch or passed individually to the model.

Let's take an example where we have multiple sequences and we pad them according to multiple objectives:


```python
sequences = [
    "I am learning machine learning",
    "I am excited to lern new frameworks in ML"
    ]

# pad sequences up to maximum sequence length
inputs = tokenizer(sequences, padding='longest')
print(inputs)
```

```
{
    'input_ids': [
            [101, 1045, 2572, 4083, 3698, 4083, 102, 0, 0, 0, 0, 0, 0], 
            [101, 1045, 2572, 7568, 2000, 3393, 6826, 2047, 7705, 2015, 1999, 19875, 102]
            ], 
    'attention_mask': [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}
```


```python
# pad sequences up to model maximum length (512 for BERT)
inputs = tokenizer(sequences, padding='max_length')
print(inputs)
```

```
{'input_ids': [[101, 1045, 2572, 4083, 3698, 4083, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1045, 2572, 7568, 2000, 3393, 6826, 2047, 7705, 2015, 1999, 19875, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
```


```python
# pad sequences up to specified maximum length
inputs = tokenizer(sequences, padding='max_length', max_length=8)
print(inputs)
```

```
{'input_ids': [[101, 1045, 2572, 4083, 3698, 4083, 102, 0], [101, 1045, 2572, 7568, 2000, 3393, 6826, 2047, 7705, 2015, 1999, 19875, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

We can also trucate sequences

```python
# truncate sequences that are longer than the model max length
inputs = tokenizer(
    sequences,
    truncation=True
    )
print(inputs)
```

```
{'input_ids': [[101, 1045, 2572, 4083, 3698, 4083, 102], [101, 1045, 2572, 7568, 2000, 3393, 6826, 2047, 7705, 2015, 1999, 19875, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

```python
# truncate sequences that are longer than the specified max length
inputs = tokenizer(
    sequences,
    max_length=8,
    truncation=True
)
print(inputs)
```

```
{'input_ids': [[101, 1045, 2572, 4083, 3698, 4083, 102], [101, 1045, 2572, 7568, 2000, 3393, 6826, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]}
```

Moreover, we can utilize the `tokenizer` object to return tensors from different frameworks, like

* `pt` return PyTorch tensors
* `tf` return Tensorflow tensors
* `np` returns NumPy tensors.


```python
# return Pytorch tensors
inputs = tokenizer(sequences, padding=True, return_tensors='pt')
print(inputs)
```

```
{'input_ids': tensor([
    [  101,  1045,  2572,  4083,  3698,  4083,   102,     0,     0,     0,     0,     0,     0],
    [  101,  1045,  2572,  7568,  2000,  3393,  6826,  2047,  7705,  2015,    1999, 19875,   102]]), 
'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

```python
# return tensorflow tensors
inputs = tokenizer(sequences, padding=True, return_tensors='tf')
print(inputs)
```

```
{'input_ids': <tf.Tensor: shape=(2, 13), dtype=int32, 
            numpy=array([[  101,  1045,  2572,  4083,  3698,  4083,   102,     0,     0,
                            0,     0,     0,     0],
                        [  101,  1045,  2572,  7568,  2000,  3393,  6826,  2047,  7705,
                            2015,  1999, 19875,   102]], dtype=int32)>, 
'attention_mask': <tf.Tensor: shape=(2, 13), dtype=int32, 
numpy=array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)>}
```


```python
# return numpy tensors
inputs = tokenizer(sequences, padding=True, return_tensors='np')
print(inputs)
```

```
{'input_ids': array([
    [  101,  1045,  2572,  4083,  3698,  4083,   102,     0,     0,    0,     0,     0,     0],
    [  101,  1045,  2572,  7568,  2000,  3393,  6826,  2047,  7705,    2015,  1999, 19875,   102]]), 
'attention_mask': array([
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

### Attention masks

**Attention masks** are defined as tensors with a value of 1s or 0s, where,

* 1s indicate the token should be attended to, and
* 0s represent the token should not be attended to.

Let's see how we use attention for our above example


```python
batch_ids = [
  [100, 100, 100],
  [100, 100, tokenizer.pad_token_id]
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0]
]
```


```python
outputs = model(
    torch.tensor(batch_ids),
    attention_mask = torch.tensor(attention_mask)
    )

print(outputs.logits)
```

```
tensor([[ 1.4738, -1.3271],
        [ 1.2205, -1.1099]], grad_fn=<AddmmBackward0>)
```

Now, we have the same logits for the second sequence as we expected to see.

### Longer sequence

Transformer models can hadle sequence length up to 512 or 1024 tokens and will crash if applied longer sequence length.

**Solution**:

* Use model with longer supported sequence length, or
* Truncate the sequence length.
