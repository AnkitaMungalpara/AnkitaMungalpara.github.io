---
title: 'LLMs and Chat Models - Hugging Face'
date: 2025-04-05

parent: Generative AI with LangChain

nav_order: 2

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---


# Accessing Open Source LLMs with Hugging Face
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!-- **Get HuggingFace Token**

```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
``` -->

## Accessing Open LLMs with Hugging Face Serverless API

The free [serverless API](https://huggingface.co/inference-api/serverless) enables quick implementation and iteration of solutions. However, it may be rate-limited for heavy use cases due to shared loads with other requests.

Here, we will use the free serverless API, which performs well in most cases. The key advantage is that we don't need to download the models or run them locally on a GPU compute infrastructure, saving both time and significant costs.

### Accessing Microsoft Phi-3 Mini Instruct

The **Phi-3-mini-4k-instruct** is a state-of -the-art, lightweight open model with 3.8 billion parameters. It has been trained using the Phi-3 datasets, which include both synthetic data and filtered publicly available websites, with a focus on high-quality and reasoning-dense content. You can find more details [here](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)


```python
from langchain_community.llms import HuggingFaceEndpoint

PHI3_ENPOINT_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

phi3_params = {
                  "wait_for_model": True,     # waits if model is not available in HuggingFace serve
                  "do_sample": False,         # temperature = 0
                  "return_full_text": False,  # will not return input prompt
                  "max_new_tokens": 1000      # max tokens answer can go upto
              }
```


```python
llm = HuggingFaceEndpoint(
                  endpoint_url = PHI3_ENPOINT_URL,
                  task = "text-generation",
                  **phi3_params
              )
```



```python
prompt = "what is generative AI in 3 points"
print(prompt)
```

```
what is generative AI in 3 points
```


```python
phi3_prompt = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
what is generative AI in 3 points<|end|>
<|assistant|>
"""

print(phi3_prompt)
```

```
<|system|>
You are a helpful assistant.<|end|>
<|user|>
what is generative AI in 3 points<|end|>
<|assistant|>
```  



```python
response = llm.invoke(phi3_prompt)
print(response)
```

```
**1. Generative AI creates new content, such as text, images, music, and code.**

**2. It utilizes machine learning algorithms to learn patterns and structures in vast datasets.**

**3. It aims to automate content creation and generate novel and original ideas.
```

### Accessing Google Gemma 2B Instruct

Gemma is a family of lightweight, state-of-the-art open models from Google, developed using the same research and technology behind the Gemini models. These text-to-text, decoder-only large language models are available in English and come with open weights, pre-trained variants, and instruction-tuned variants.

Gemma models are ideal for various text generation tasks, including question-answering, summarization, and reasoning. For more details, check [here](https://huggingface.co/google/gemma-1.1-2b-it).


```python
GEMMA_ENDPOINT_URL = "https://api-inference.huggingface.co/models/google/gemma-1.1-2b-it"

gemma_params = {
                  "wait_for_model": True,     # waits if model is not available in HuggingFace serve
                  "do_sample": False,         # temperature = 0
                  "return_full_text": False,  # will not return input prompt
                  "max_new_tokens": 1000      # max tokens answer can go upto
              }
```


```python
llm = HuggingFaceEndpoint(
                endpoint_url = GEMMA_ENDPOINT_URL,
                task = "text-generation",
                **gemma_params
            )
```




```python
response_gemma = llm.invoke(prompt)
print(response_gemma)
```

``` 
1. Generative AI models create new content, such as text, images, and videos, based on existing data.

2. These models are trained on large datasets of text, images, and other media.

3. Generative AI models can produce novel and creative content, often surpassing human-level performance in specific tasks.
```

## Accessing Local LLMs with HuggingFacePipeline API


Hugging Face models can be run locally using the `HuggingFacePipeline` class, but a good GPU is needed for fast inference. The Hugging Face Model Hub hosts over 500k models, including 90K+ open LLMs. These models can be accessed through LangChain either via the local pipeline wrapper or by using the hosted inference endpoints through the `HuggingFaceEndpoint` API.

To use these models, ensure we have the transformers and pytorch packages installed. The main advantage of running models locally is the enhanced privacy and security. However, it requires a robust compute infrastructure, preferably with a GPU.

### Accessing Google Gemma 2B and running it locally


```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

gemma_params = {
                  "do_sample": False,         # temperature = 0
                  "return_full_text": False,  # will not return input prompt
                  "max_new_tokens": 1000      # max tokens answer can go upto
              }
```


```python
gemma_local_llm = HuggingFacePipeline.from_model_id(
                              model_id="google/gemma-1.1-2b-it",
                              task = "text-generation",
                              pipeline_kwargs = gemma_params,
                              device = 0        # when running on Colab it selects GPU
                          )
print(gemma_local_llm)
```




```
HuggingFacePipeline(pipeline=<transformers.pipelines.text_generation.TextGenerationPipeline object>, model_id='google/gemma-1.1-2b-it', model_kwargs={}, pipeline_kwargs={'do_sample': False, 'return_full_text': False, 'max_new_tokens': 1000})
```



```python
print(prompt)
```
```
'what is generative AI in 3 points'
```



```python
gemma_prompt = """<bos><start_of_turn>user
what is generative AI in 3 points<end_of_turn>
<start_of_turn>model"""

print(gemma_prompt)
```

```
<bos><start_of_turn>user
what is generative AI in 3 points<end_of_turn>
<start_of_turn>model
```


```python
response = gemma_local_llm.invoke(gemma_prompt)
print(response)
```

```
1. **Generative AI creates new content:** It uses algorithms to analyze vast datasets and learn patterns, enabling it to generate novel text, images, music, videos, and other forms of media.

2. **Powered by deep learning:** Generative AI models are built using deep learning techniques, which allow them to learn complex relationships from vast amounts of data.

3. **Focuses on both content creation and understanding:** Generative AI aims to both generate new content and also understand the underlying patterns and structures within existing data.
```


## Accessing Open LLMs with HuggingFace as a Chat Model LLM

Here, we will demonstrate how to access open LLMs from HuggingFace, such as Google's Gemma 2B.


```python
from langchain_community.chat_models import ChatHuggingFace

gemma_chat = ChatHuggingFace(
                      llm = llm,
                      model_id = "google/gemma-1.1-2b-it"
                  )
```


```python
response_gemma_chat = gemma_chat.invoke(prompt)
print(response_gemma_chat)
```

```
AIMessage(content='1. **Generative AI** refers to the creation of new content, such as text, images, videos, music, or code, based on existing data or instructions.\n\n\n2. **Key techniques** include deep learning algorithms and statistical models that learn patterns and relationships from vast datasets.\n\n\n3. **Goal** is to automate content creation, enhance productivity, and provide personalized experiences by generating novel and relevant outputs.')
```



```python
print(response_gemma_chat.content)
```

```
1. **Generative AI** refers to the creation of new content, such as text, images, videos, music, or code, based on existing data or instructions.

2. **Key techniques** include deep learning algorithms and statistical models that learn patterns and relationships from vast datasets.

3. **Goal** is to automate content creation, enhance productivity, and provide personalized experiences by generating novel and relevant outputs.
```
