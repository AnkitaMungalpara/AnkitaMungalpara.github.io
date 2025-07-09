---
title: 'LLMs and Chat Models - OpenAI'
date: 2025-04-05

parent: Generative AI with LangChain

nav_order: 1

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# LLMs and Chat Models
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

**Importing Required Dependencies**


```python
import os
from google.colab import userdata
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
```

## Model Input/Output

In LangChain, the language model is the core of any application, ensuring smooth integration and communication.

### Key Components of Model Input/Output: LLMs and Chat Models

- **LLMs:** Pure text completion models that take a text string as input nd return a text string.

- **Chat Models:** Language models with different I/O types, taking a list of chat messages as input and producing a chat message as output.


## Accessing Commercial LLMs like ChatGPT as an Instruct LLM


```python
# load the Open AI model
llm = OpenAI(
    model_name = "gpt-3.5-turbo-instruct",  # specify model name to use
    temperature=0,    # set the temperature (0 means less randomness)
)
```


```python
# define prompt
prompt = "Explain Artificial Intelligence in three points"
print(prompt)
```

```
Explain Artificial Intelligence in three points
```


```python
# invoke the model with defined prompt
response = llm.invoke(prompt)
print(response)
```

```
1. Artificial Intelligence (AI) is a branch of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence. This includes tasks such as learning, problem-solving, decision making, and natural language processing.

2. AI systems use algorithms and data to analyze and learn from patterns and make predictions or decisions based on that information. This allows them to continuously improve and adapt to new situations, making them more efficient and accurate over time.

3. AI has a wide range of applications, including virtual assistants, self-driving cars, medical diagnosis, and fraud detection. It has the potential to greatly impact various industries and improve our daily lives by automating tasks, increasing efficiency, and providing valuable insights. However, there are also concerns about the ethical implications and potential risks of AI, such as job displacement and biased decision-making. 
```

## Accessing ChatGPT as a Chat Model LLM


```python
# load Chat Open AI model with below parameters
llm = ChatOpenAI(
    model_name = "gpt-3.5-turbo",
    temperature=0
)
```

```python
prompt = "Explain Artificial Intelligence in three points"
print(prompt)
```

```
Explain Artificial Intelligence in three points
```


```python
# get response by invoking the llm model with defined prompt
response = llm.invoke(prompt)
response
```



```
AIMessage(content='1. Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines, such as learning, reasoning, problem-solving, perception, and language understanding. AI systems are designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.\n\n2. AI technologies include machine learning, neural networks, natural language processing, computer vision, and robotics. These technologies enable machines to analyze and interpret complex data, make decisions based on patterns and trends, and interact with humans in a more natural and intuitive way.\n\n3. AI has a wide range of applications across various industries, including healthcare, finance, transportation, retail, and entertainment. AI systems are used to improve efficiency, accuracy, and productivity in tasks such as diagnosing diseases, predicting market trends, optimizing supply chains, and personalizing customer experiences. However, AI also raises ethical and societal concerns, such as job displacement, privacy issues, and bias in decision-making.', response_metadata={'token_usage': {'completion_tokens': 196, 'prompt_tokens': 14, 'total_tokens': 210}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None},)
```


```python
# print response content
print(response.content)
```

```
1. Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines, such as learning, reasoning, problem-solving, perception, and language understanding. AI systems are designed to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

2. AI technologies include machine learning, neural networks, natural language processing, computer vision, and robotics. These technologies enable machines to analyze and interpret complex data, make decisions based on patterns and trends, and interact with humans in a more natural and intuitive way.

3. AI has a wide range of applications across various industries, including healthcare, finance, transportation, retail, and entertainment. AI systems are used to improve efficiency, accuracy, and productivity in tasks such as diagnosing diseases, predicting market trends, optimizing supply chains, and personalizing customer experiences. However, AI also raises ethical and societal concerns, such as job displacement, privacy issues, and bias in decision-making.
```
