---
title: 'Conversational Prompting with OpenAI and Hugging Face'
date: 2025-04-05

parent: Generative AI with LangChain

nav_order: 3

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---


# Message Types for Chat Models and Conversational Prompting
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

Conversational prompting involves engaging in a conversation with a Large Language Model (LLM), represented as a list of messages. Each message has the following properties:

*   **Role:** Indicates the speaker. LangChain provides different message classes for various roles.

*   **Content:** The message substance, either a string or a list of dictionaries for multi-modal inputs.

*   **Additional_kwargs:** Used for extra information specific to the message provider, like `function_call` in OpenAI.


## Message Types


*   **HumanMessage:** A user-generated message, usually just content.

*   **AIMessage:** A model-generated message, potentially including `additional_kwargs` like `tool_calls`.

*   **SystemMessage:** A system instruction for the model's behavior, typically just content. Not all models support this type.


<!-- **Get OpenAI API Tokens**


```python
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
``` -->

## Conversational Prompting with ChatGPT


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
          model_name = "gpt-3.5-turbo",
          temperature=0
      )
```


```python
from langchain_core.messages import HumanMessage, SystemMessage

prompt = """What is Deep Learning in 3 points?"""
system_prompt = """Act as a helpful assistant and give meaningful examples in your responses."""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=prompt)
]
print(messages)
```

```
[SystemMessage(content='Act as a helpful assistant and give meaningful examples in your responses.'),
    HumanMessage(content='What is Deep Learning in 3 points?')]
```



```python
response = llm.invoke(messages)
print(response)
```


```
AIMessage(content='1. Deep learning is a subset of machine learning that involves training artificial neural networks with multiple layers to learn and make decisions from large amounts of data. For example, deep learning is used in image recognition tasks, where a neural network learns to identify objects in images by analyzing patterns in the pixel data.\n\n2. Deep learning models are capable of automatically learning hierarchical representations of data, which allows them to extract complex features and patterns from raw input. For instance, in natural language processing, deep learning models can learn to understand and generate human-like text by processing and analyzing sequences of words.\n\n3. Deep learning has been successfully applied in various fields such as computer vision, speech recognition, and natural language processing. For example, deep learning models have been used to develop self-driving cars that can perceive and navigate through their environment, as well as to create virtual assistants like Siri and Alexa that can understand and respond to human language.', response_metadata={'token_usage': {'completion_tokens': 183, 'prompt_tokens': 33, 'total_tokens': 216}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None})
```



```python
print(response.content)
```
```
1. Deep learning is a subset of machine learning that involves training artificial neural networks with multiple layers to learn and make decisions from large amounts of data. For example, deep learning is used in image recognition tasks, where a neural network learns to identify objects in images by analyzing patterns in the pixel data.
2. Deep learning models are capable of automatically learning hierarchical representations of data, which allows them to extract complex features and patterns from raw input. For instance, in natural language processing, deep learning models can learn to understand and generate human-like text by processing and analyzing sequences of words.
3. Deep learning has been successfully applied in various fields such as computer vision, speech recognition, and natural language processing. For example, deep learning models have been used to develop self-driving cars that can perceive and navigate through their environment, as well as to create virtual assistants like Siri and Alexa that can understand and respond to human language.
```


```python
# add the past conversation history into messages
messages.append(response)

# add the new prompt to the conversation history list
prompt = """What did we talk about?"""
messages.append(
            HumanMessage(content=prompt)
          )
print(messages)
```

```
[SystemMessage(content='Act as a helpful assistant and give meaningful examples in your responses.'),
    HumanMessage(content='What is Deep Learning in 3 points?'),
    AIMessage(content='1. Deep learning is a subset of machine learning that involves training artificial neural networks with multiple layers to learn and make decisions from large amounts of data. For example, deep learning is used in image recognition tasks, where a neural network learns to identify objects in images by analyzing patterns in the pixel data.\n\n2. Deep learning models are capable of automatically learning hierarchical representations of data, which allows them to extract complex features and patterns from raw input. For instance, in natural language processing, deep learning models can learn to understand and generate human-like text by processing and analyzing sequences of words.\n\n3. Deep learning has been successfully applied in various fields such as computer vision, speech recognition, and natural language processing. For example, deep learning models have been used to develop self-driving cars that can perceive and navigate through their environment, as well as to create virtual assistants like Siri and Alexa that can understand and respond to human language.', response_metadata={'token_usage': {'completion_tokens': 183, 'prompt_tokens': 33, 'total_tokens': 216}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}),
    HumanMessage(content='What did we talk about?')]
```



```python
# send the conversation history along with the new prompt to the llm
response = llm.invoke(messages)
print(response.content)
```

```
We discussed the concept of deep learning in three key points:

1. Deep learning as a subset of machine learning that involves training artificial neural networks with multiple layers.
2. The ability of deep learning models to automatically learn hierarchical representations of data to extract complex features and patterns.
3. Examples of applications of deep learning in fields such as computer vision, speech recognition, and natural language processing.
```

## Conversational Prompting with Open LLMs via HuggingFace

Using the `ChatModel` API in `ChatHuggingFace`, we can have full conversations with open LLMs, maintaining the historical flow of the conversation. For example, here we are using the **Google Gemma 2B LLM**.

<!-- **Get HuggingFace API Token**

```python
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
``` -->

```python
from langchain_community.llms import HuggingFaceEndpoint
```


```python
GEMMA_API_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it"

gemma_params = {
    "wait_for_model": True,
    "do_sample": False,
    "return_full_text": False,
    "max_new_tokens": 1000
}
```


```python
gemma_llm = HuggingFaceEndpoint(
    endpoint_url = GEMMA_API_URL,
    task = "text-generation",
    **gemma_params
)
```



```python
from ast import mod
from langchain_community.chat_models import ChatHuggingFace

gemma_chat = ChatHuggingFace(
                      llm = gemma_llm,
                      model_id = "google/gemma-1.1-2b-it"
                  )
```


```python
prompt = """Explain Machine Learning in 3 points?"""
messages = [HumanMessage(content=prompt)]
```


```python
response = gemma_chat.invoke(messages)
print(response.content)
```

```
1. **Learning from data:** Machine learning algorithms are trained on large datasets to identify patterns and relationships, enabling them to make predictions, classify data, and generate new insights.
2. **Automated decision-making:** By analyzing vast amounts of data, machine learning models can learn complex patterns and make decisions without human intervention, freeing up human resources for higher-level tasks.
3. **Continual improvement:** Machine learning models are constantly updated and refined as new data is collected, allowing them to improve their performance over time and adapt to new situations.
```


```python
messages.append(response)
print(messages)
```

```
[HumanMessage(content='Explain Machine Learning in 3 points?'),
    AIMessage(content='1. **Learning from data:** Machine learning algorithms are trained on large datasets to identify patterns and relationships, enabling them to make predictions, classify data, and generate new insights.\n\n\n2. **Automated decision-making:** By analyzing vast amounts of data, machine learning models can learn complex patterns and make decisions without human intervention, freeing up human resources for higher-level tasks.\n\n\n3. **Continual improvement:** Machine learning models are constantly updated and refined as new data is collected, allowing them to improve their performance over time and adapt to new situations.')]
```



```python
print(gemma_chat._to_chat_prompt([messages[0]]))
```
```
<bos><start_of_turn>user
Explain Machine Learning in 3 points?<end_of_turn>
<start_of_turn>model
```



```python
prompt = """Now do the same for Deep Learning"""
messages.append(
    HumanMessage(content=prompt)
  )
```


```python
response = gemma_chat.invoke(messages)
print(response.content)
```

```
**3 Points about Deep Learning:**

1. **Harnessing the power of artificial neural networks:** Deep learning builds upon traditional machine learning techniques by employing deep neural networks with multiple layers to extract complex patterns and make intricate predictions.
2. **Learning from vast data:** Deep learning algorithms are trained on massive datasets of structured and unstructured data to learn intricate relationships and make predictions on new data.
3. **Automated feature extraction and learning:** Deep learning models are capable of automatically extracting meaningful features from large datasets, reducing the need for manual feature engineering, and enabling faster and more accurate learning.
```
