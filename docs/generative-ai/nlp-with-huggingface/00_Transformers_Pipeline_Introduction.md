---
title: 'Transformers Pipelines Introduction'
date: 2025-04-05

parent: NLP with Hugging Face

nav_order: 1

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Pipelines
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

The pipelines are an efficient way to utilize models for inference. These pipelines are objects that abstract the majority of complex codebase from the library, providing a simple API dedicated to multiple tasks, including sentiment analysis, named entity recognition, summarization, text generation, and question answering.


```python
from transformers import pipeline
```


## Sentiment Analysis - Classification

### Passing one sentence


```python
classifier = pipeline('sentiment-analysis')
classifier('I have been waiting to start Hugging Face course')
```

```
[{'label': 'POSITIVE', 'score': 0.9987032413482666}]
```


### Passing multiple sentences


```python
classifier = pipeline('sentiment-analysis')
classifier(
    ['I have been waiting to start Hugging Face course',
     'That is such a bad news!'])
```

```
[{'label': 'POSITIVE', 'score': 0.9987032413482666},
    {'label': 'NEGATIVE', 'score': 0.9998062252998352}]
```


## Zero-shot Classification


```python
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about Natural language processing (NLP)",
    candidate_labels = ["education", "sports", "entertainment"]
)
```

```
{'sequence': 'This is a course about Natural language processing (NLP)',
    'labels': ['education', 'entertainment', 'sports'],
    'scores': [0.614361047744751, 0.2008926123380661, 0.18474635481834412]}
```

## Text generation


```python
generator = pipeline('text-generation')
generator("In this algorithm you will learn")
```

```
[{'generated_text': 'In this algorithm you will learn how to store a random number within a string, then compare the result against the array of strings stored in the array and assign a single token to the key of the token in the first key string in the array.\n'}]
```



```python
generator = pipeline('text-generation', model = 'distilgpt2')
generator(
    "In this algortihm you will learn",
    max_length = 50,
    num_return_sequences = 3
)
```
```
[{'generated_text': 'In this algortihm you will learn from your past and the future. The future is never to come without the knowledge of your future and from that present.\n\nYou must also follow me on Twitter: @jeffdew_'},
    {'generated_text': "In this algortihm you will learn to read as well as learn how to read!\n\nSo, once you read this, here's my first question.\n1) I will have to ask you to use your word correctly by"},
    {'generated_text': 'In this algortihm you will learn a lot about the way animals deal with the environment in a food-loving world. For example, how are animals able to handle the environment in which animals live?\n\n\n\nMany animals ('}]
```

## Fill Mask


```python
fill_masker = pipeline('fill-mask')
fill_masker(
    "From this course, you will learn all about <mask> methods.",
    top_k = 3
)
```

```
[{'score': 0.042266182601451874,
'token': 8326,
'token_str': ' programming',
'sequence': 'From this course, you will learn all about programming methods.'},
{'score': 0.017150819301605225,
'token': 25212,
'token_str': ' optimization',
'sequence': 'From this course, you will learn all about optimization methods.'},
{'score': 0.016385400667786598,
'token': 17325,
'token_str': ' statistical',
'sequence': 'From this course, you will learn all about statistical methods.'}]
```


## Named Entity Recognition (NER)


```python
ner = pipeline('ner',
               grouped_entities = True
              )
ner("My name is John and I work at Amazon in Seattle.")
```

```

[{'entity_group': 'PER',
'score': 0.9986583,
'word': 'John',
'start': 11,
'end': 15},
{'entity_group': 'ORG',
'score': 0.99745816,
'word': 'Amazon',
'start': 30,
'end': 36},
{'entity_group': 'LOC',
'score': 0.99858034,
'word': 'Seattle',
'start': 40,
'end': 47}]
```



## Question Answering


```python
que_answer = pipeline('question-answering')
que_answer(
    question = 'what is my name?',
    context='My name is John and I work at Amazon in Seattle.'
)
```

```
{'score': 0.9963988065719604, 'start': 11, 'end': 15, 'answer': 'John'}
```


## Summarization


```python
summarizer = pipeline('summarization')
summarizer(
    """
    Us is a 2019 psychological horror film written and directed by Jordan Peele, starring Lupita Nyong'o, Winston Duke, Elisabeth Moss, and Tim Heidecker.
    The film follows Adelaide Wilson (Nyong'o) and her family, who are attacked by a group of menacing doppelgängers, called the ‘Tethered’. The project was announced in February 2018, and much of the cast joined in the following months. Peele produced the film alongside Jason Blum and Sean McKittrick, having previously collaborated on Get Out and BlacKkKlansman, as well as Ian Cooper. Filming took place in California, mostly in Los Angeles, Pasadena and Santa Cruz, from July to October 2018.
    Us premiered at South by Southwest on March 8, 2019, and was theatrically released in the United States on March 22, 2019, by Universal Pictures. It was a critical and commercial success, grossing $256 million worldwide against a budget of $20 million, and received praise for Peele's screenplay and direction, Nyong'o's performance, and Michael Abels' score.
    """
)
```

```
[{'summary_text': " Us premiered at South by Southwest on March 8, 2019, and was theatrically released in the U.S. on March 22, 2019 . It was a critical and commercial success, grossing $256 million worldwide against a budget of $20 million . The film follows Adelaide Wilson (Nyong'o) and her family, who are attacked by a group of menacing doppelgängers ."}]
```



```python
summarizer = pipeline('summarization')
summarizer(
    """
    Us is a 2019 psychological horror film written and directed by Jordan Peele, starring Lupita Nyong'o, Winston Duke, Elisabeth Moss, and Tim Heidecker.
    The film follows Adelaide Wilson (Nyong'o) and her family, who are attacked by a group of menacing doppelgängers, called the ‘Tethered’. The project was announced in February 2018, and much of the cast joined in the following months. Peele produced the film alongside Jason Blum and Sean McKittrick, having previously collaborated on Get Out and BlacKkKlansman, as well as Ian Cooper. Filming took place in California, mostly in Los Angeles, Pasadena and Santa Cruz, from July to October 2018.
    Us premiered at South by Southwest on March 8, 2019, and was theatrically released in the United States on March 22, 2019, by Universal Pictures. It was a critical and commercial success, grossing $256 million worldwide against a budget of $20 million, and received praise for Peele's screenplay and direction, Nyong'o's performance, and Michael Abels' score.
    """,
    min_length = 50,
    max_length = 200
)
```

```
[{'summary_text': " Us was written and directed by Jordan Peele, starring Lupita Nyong'o, Winston Duke, Elisabeth Moss, and Tim Heidecker . The film follows Adelaide Wilson and her family, who are attacked by a group of menacing doppelgängers . It was a critical and commercial success, grossing $256 million worldwide against a budget of $20 million ."}]
```


## Translation


```python
translator = pipeline('translation',
                      model = 'Helsinki-NLP/opus-mt-en-fr'
                    )
translator('I am learning Natural Language Processing (NLP) course.')

```

```
[{'translation_text': "J'apprends le cours de traitement des langues naturelles (NLP)."}]
```

