---
title: 'Fine-tuning Model with Trainer API'
date: 2025-04-05

parent: NLP with Hugging Face

nav_order: 6

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Fine-tuning Model with Trainer API
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

Transformers provides a `Trainer` object to help in fine-tuning any of the pretrained models it provides on your dataset.


```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_dataset = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
```

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

```python
def tokenize_function(sample):
  return tokenizer(sample["sentence1"], sample["sentence2"], truncation=True)
```

```python
tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```


## Training


```python
from transformers import TrainingArguments
training_args = TrainingArguments("trainer")
```

Defining the model


```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Now, we define `Trainer` by passing all the objects constructed up to now.


```python
from transformers import Trainer

trainer = Trainer(
  model,
  training_args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["validation"],
  data_collator=data_collator,
  tokenizer=tokenizer
)
trainer.train()
```


<div style="margin-bottom: 20px;">
  <progress value="1377" max="1377" style="width: 300px; height: 20px; vertical-align: middle;"></progress>
  <span style="margin-left: 10px;">[1377/1377 03:53, Epoch 3/3]</span>
</div>

<table class="dataframe" style="border-collapse: collapse; width: 50px; text-align: center;">
  <thead>
    <tr>
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>0.506700</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.249000</td>
    </tr>
  </tbody>
</table>

```
TrainOutput(global_step=1377, training_loss=0.30210025984044514, metrics={'train_runtime': 236.2031, 'train_samples_per_second': 46.587, 'train_steps_per_second': 5.83, 'total_flos': 405114969714960.0, 'train_loss': 0.30210025984044514, 'epoch': 3.0})
```


## Evaluation

Now, we'll create predictions for the model we built, and for that, we will use the `Trainer.predict()` command.


```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

```
(408, 2) (408,)
```

The output of the `predict()` method is a named tuple containing three fields:
* **predictions**
* **label_ids**, and
* **metrics**

We can see from the above output that we need to convert the logits returned by the model to transform them into predictions.


```python
import numpy as np
preds = np.argmax(predictions.predictions, axis=-1)
```

It's time to evaluate the model


```python
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(
    predictions=preds,
    references=predictions.label_ids
)
```

```
{'accuracy': 0.8480392156862745, 'f1': 0.8949152542372881}
```


Here's time to wrap everything together.


```python
def compute_metrics(eval_predictions):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_predictions
    preds = np.argmax(logits, axis=-1)

    return metric.compute(
        predictions=preds,
        references=labels
  )
```

Now, we'll define `Trainer` with functionality of reporting metrucs at the end of each epoch.

```python
training_args = TrainingArguments("trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

```python
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

Now that we have defined the `Trainer` with passing additional arguments of evaluation strategy and compute_metric, we execute new training.


```python
trainer.train()
```

<div style="margin-bottom: 20px;">
  <progress value="1377" max="1377" style="width: 300px; height: 20px; vertical-align: middle;"></progress>
  <span style="margin-left: 10px;">[1377/1377 03:50, Epoch 3/3]</span>
</div>

<table class="dataframe" style="border-collapse: collapse; width: 100%; text-align: center;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.417862</td>
      <td>0.821078</td>
      <td>0.875639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.542100</td>
      <td>0.450163</td>
      <td>0.845588</td>
      <td>0.891566</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.340800</td>
      <td>0.662934</td>
      <td>0.843137</td>
      <td>0.889273</td>
    </tr>
  </tbody>
</table>

```
TrainOutput(global_step=1377, training_loss=0.3811595143835529, metrics={'train_runtime': 230.8587, 'train_samples_per_second': 47.666, 'train_steps_per_second': 5.965, 'total_flos': 405114969714960.0, 'train_loss': 0.3811595143835529, 'epoch': 3.0})
```

