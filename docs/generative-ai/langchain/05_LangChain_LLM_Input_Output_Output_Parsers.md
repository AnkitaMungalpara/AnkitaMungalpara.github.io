---
title: 'Output Parsers'
date: 2025-04-05

parent: Generative AI with LangChain

nav_order: 5

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# LLM Input/Output - Output Parsers
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!-- **Enter API Tokens**

```python
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
``` -->

**Chat Models and LLMs**

LangChain uses existing Large Language Models (LLMs) from various providers like OpenAI and Hugging Face. It does not build its own LLMs but offers a standard API to interact with different LLMs through a standard interface.

**Accessing Commercial LLMs like ChatGPT**


```python
from langchain_openai import ChatOpenAI

# instantiate the model
llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0
    )
```

## Output Parsers

Output parsers in Langchain are crucial for structuring responses from language models. Here are examples of Langchain's specific parser types:

*   **PydanticOutputParser**:
    - Uses Pydantic models to ensure outputs match a specified schema, providing type checking and coercion similar to Python dataclasses.

*   **JsonOutputParser**:
    - Ensures outputs adhere to an arbitrary JSON schema, with Pydantic models optionally used to declare the data structure.

*   **CommaSeparatedListOutputParser**:
    - Extracts comma-separated values from model outputs, useful for lists of items.


### Pydantic OutputParser


```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
```


```python
# define the desired data structure
class QueryResponse(BaseModel):
  description: str = Field(description= "A brief description of the topic asked by the user")
  pros: str = Field(description='three points showing the pros of the topic asked by the user')
  cons: str = Field(description='three points showing the cons of the topic asked by the user')
  conclusion: str = Field(description="summary of topic asked by the user")

# Set up a parser and add instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=QueryResponse)
print(parser)
```

```
PydanticOutputParser(pydantic_object=<class '__main__.QueryResponse'>)
```

```python
print(parser.get_format_instructions())
```

```
The output should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema 
{"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
{"properties": {"description": {"title": "Description", "description": "A brief description of the topic asked by the user", "type": "string"}, "pros": {"title": "Pros", "description": "three points showing the pros of the topic asked by the user", "type": "string"}, "cons": {"title": "Cons", "description": "three points showing the cons of the topic asked by the user", "type": "string"}, "conclusion": {"title": "Conclusion", "description": "summary of topic asked by the user", "type": "string"}}, "required": ["description", "pros", "cons", "conclusion"]}
```


```python
# create final prompt with formatting instructions from the parser

prompt_txt = """
    Answer the user query and generate the response based on the following formmatted instructions:

    formatted instructions:
    {format_instructions}

    Query:
    {query}
  """
```

```python
prompt = PromptTemplate(
    template=prompt_txt,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
print(prompt)
```

```
PromptTemplate(input_variables=['query'], partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"description": {"title": "Description", "description": "A brief description of the topic asked by the user", "type": "string"}, "pros": {"title": "Pros", "description": "three points showing the pros of the topic asked by the user", "type": "string"}, "cons": {"title": "Cons", "description": "three points showing the cons of the topic asked by the user", "type": "string"}, "conclusion": {"title": "Conclusion", "description": "summary of topic asked by the user", "type": "string"}}, "required": ["description", "pros", "cons", "conclusion"]}\n```'}, template='\n              Answer the user query and generate the response based on the following formmatted instructions:\n\n              formatted instructions:\n              {format_instructions}\n\n              Query:\n              {query}\n            ')
```



```python
chain = (prompt | llm | parser)
```


```python
question = "Tell me about the carbon sequestration"
# invoke chain
response = chain.invoke({"query": question})
# get the response
print(response)
```


```
QueryResponse(description='Carbon sequestration is the process of capturing and storing carbon dioxide to mitigate its presence in the atmosphere and reduce the impact of climate change.', pros='1. Helps reduce greenhouse gas emissions. 2. Can help restore degraded lands. 3. Provides economic opportunities in carbon offset markets.', cons='1. Requires significant investment and technology. 2. Long-term storage risks and uncertainties. 3. Potential for negative environmental impacts if not managed properly.', conclusion='Overall, carbon sequestration has the potential to play a significant role in addressing climate change, but careful planning and monitoring are essential to ensure its effectiveness and sustainability.')
```



```python
print(response.description)
```

```
'Carbon sequestration is the process of capturing and storing carbon dioxide to mitigate its presence in the atmosphere and reduce the impact of climate change.'
```

```python
# printing as dictionary
response.dict()
```


```
{'description': 'Carbon sequestration is the process of capturing and storing carbon dioxide to mitigate its presence in the atmosphere and reduce the impact of climate change.',
'pros': '1. Helps reduce greenhouse gas emissions. 2. Can help restore degraded lands. 3. Provides economic opportunities in carbon offset markets.',
'cons': '1. Requires significant investment and technology. 2. Long-term storage risks and uncertainties. 3. Potential for negative environmental impacts if not managed properly.',
'conclusion': 'Overall, carbon sequestration has the potential to play a significant role in addressing climate change, but careful planning and monitoring are essential to ensure its effectiveness and sustainability.'}
```



```python
for key, value in response.dict().items():
  print(f"{key}:\n{value}\n")
```

```
description:
Carbon sequestration is the process of capturing and storing carbon dioxide to mitigate its presence in the atmosphere and reduce the impact of climate change.

pros:
1. Helps reduce greenhouse gas emissions. 2. Can help restore degraded lands. 3. Provides economic opportunities in carbon offset markets.

cons:
1. Requires significant investment and technology. 2. Long-term storage risks and uncertainties. 3. Potential for negative environmental impacts if not managed properly.

conclusion:
Overall, carbon sequestration has the potential to play a significant role in addressing climate change, but careful planning and monitoring are essential to ensure its effectiveness and sustainability.
```


### JsonOutputParser


```python
from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
```


```python
# define the data structure
class QueryResponse(BaseModel):
  description: str = Field(description= "A brief description of the topic asked by the user")
  pros: str = Field(description='three points showing the pros of the topic asked by the user')
  cons: str = Field(description='three points showing the cons of the topic asked by the user')
  conclusion: str = Field(description="summary of topic asked by the user")

# set up parser
parser = JsonOutputParser(pydantic_object=QueryResponse)
print(parser)
```

```
JsonOutputParser(pydantic_object=<class '__main__.QueryResponse'>)
```



```python
# create final prompt with formatting instructions from the parser

prompt_txt = """
  Answer the user query and generate the response based on the following formmatted instructions:

  formatted instructions:
  {format_instructions}

  Query:
  {query}
"""
```


```python
# create a template for a string prompt
prompt = PromptTemplate(
    template=prompt_txt,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
print(prompt)
```


```
PromptTemplate(input_variables=['query'], partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"description": {"title": "Description", "description": "A brief description of the topic asked by the user", "type": "string"}, "pros": {"title": "Pros", "description": "three points showing the pros of the topic asked by the user", "type": "string"}, "cons": {"title": "Cons", "description": "three points showing the cons of the topic asked by the user", "type": "string"}, "conclusion": {"title": "Conclusion", "description": "summary of topic asked by the user", "type": "string"}}, "required": ["description", "pros", "cons", "conclusion"]}\n```'}, template='\n              Answer the user query and generate the response based on the following formmatted instructions:\n\n              formatted instructions:\n              {format_instructions}\n\n              Query:\n              {query}\n            ')
```



```python
# create a chain
chain = (prompt | llm | parser)
print(chain)
```

```
PromptTemplate(input_variables=['query'], partial_variables={'format_instructions': 'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"description": {"title": "Description", "description": "A brief description of the topic asked by the user", "type": "string"}, "pros": {"title": "Pros", "description": "three points showing the pros of the topic asked by the user", "type": "string"}, "cons": {"title": "Cons", "description": "three points showing the cons of the topic asked by the user", "type": "string"}, "conclusion": {"title": "Conclusion", "description": "summary of topic asked by the user", "type": "string"}}, "required": ["description", "pros", "cons", "conclusion"]}\n```'}, template='\n              Answer the user query and generate the response based on the following formmatted instructions:\n\n              formatted instructions:\n              {format_instructions}\n\n              Query:\n              {query}\n            ')
| ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x7d2609ab6fe0>, async_client=<openai.resources.chat.completions.AsyncCompletions>, temperature=0.0)
| JsonOutputParser(pydantic_object=<class '__main__.QueryResponse'>)
```



```python
queries = [
  "Tell me about the carbon sequestration",
  "Tell me about backpropagation algorithm in machine learning"
]
```


```python
queries_formatted = [{"query": subject} for subject in queries]
print(queries_formatted)
```


```
[{'query': 'Tell me about the carbon sequestration'},
{'query': 'Tell me about backpropagation algorithm in machine learning'}]
```



```python
# get the response
responses = chain.map().invoke(queries_formatted)
```


```python
import pandas as pd
# convert response to DataFrame
data = pd.DataFrame(responses)
print(data)
```


  <div id="df-e3aade0f-5749-47e6-ac56-6f755ddf30f8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>description</th>
      <th>pros</th>
      <th>cons</th>
      <th>conclusion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Carbon sequestration is the process of capturi...</td>
      <td>1. Helps reduce greenhouse gas emissions. 2. C...</td>
      <td>1. Requires significant investment and technol...</td>
      <td>Carbon sequestration has the potential to play...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Backpropagation is a key algorithm used in tra...</td>
      <td>1. Backpropagation allows neural networks to l...</td>
      <td>1. Backpropagation can suffer from the vanishi...</td>
      <td>In conclusion, backpropagation is a powerful a...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e3aade0f-5749-47e6-ac56-6f755ddf30f8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e3aade0f-5749-47e6-ac56-6f755ddf30f8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e3aade0f-5749-47e6-ac56-6f755ddf30f8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-264bd025-0739-4c8a-a61f-49bb5591b6f4">
  <button class="colab-df-quickchart" onclick="quickchart('df-264bd025-0739-4c8a-a61f-49bb5591b6f4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-264bd025-0739-4c8a-a61f-49bb5591b6f4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_725a59a1-9965-4420-8322-a4091d64ff88">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_725a59a1-9965-4420-8322-a4091d64ff88 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('data');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
for response in responses:
  for key, val in response.items():
    print(f"{key}:\n{val}\n")
  print('-----------------------------------------------------------')
```

```
description:
Carbon sequestration is the process of capturing and storing carbon dioxide to mitigate its presence in the atmosphere and combat climate change.

pros:
1. Helps reduce greenhouse gas emissions. 2. Can help improve soil quality. 3. Provides potential economic opportunities in carbon trading.

cons:
1. Requires significant investment and technology. 2. Some methods may have limited effectiveness. 3. Long-term storage risks and uncertainties.

conclusion:
Carbon sequestration has the potential to play a significant role in addressing climate change, but it also comes with challenges and uncertainties that need to be carefully considered.
-----------------------------------------------------------
description:
Backpropagation is a key algorithm used in training artificial neural networks in machine learning. It involves calculating the gradient of a loss function with respect to the weights of the network, and then using this gradient to update the weights in order to minimize the loss.

pros:
1. Backpropagation allows neural networks to learn complex patterns and relationships in data. 2. It is an efficient way to optimize the weights of a neural network. 3. Backpropagation can be used in various types of neural networks, such as feedforward and recurrent networks.

cons:
1. Backpropagation can suffer from the vanishing gradient problem in deep neural networks. 2. It requires a large amount of labeled training data to perform well. 3. Backpropagation can be computationally expensive, especially for large networks.

conclusion:
In conclusion, backpropagation is a powerful algorithm that has been instrumental in the success of neural networks in machine learning, despite some limitations.
-----------------------------------------------------------
```

### CommaSeparatedListOutputParser


```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
```


```python
# output parser
output_parser = CommaSeparatedListOutputParser()

# get formatted instructions
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
```

```
'Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`'
```



```python
# create final prompt with formatting instructions from the parser

prompt_txt = """
  List 5 real-world use cases where object detection can be used:

  output format instructions:
  {format_instructions}
"""
```


```python
prompt = PromptTemplate.from_template(template=prompt_txt)
print(prompt)
```

```
PromptTemplate(input_variables=['format_instructions'], template='\n              List 5 real-world use cases where object detection can be used:\n\n              output format instructions:\n              {format_instructions}\n\n            ')
```

```python
chain = (prompt | llm | output_parser)
```


```python
response = chain.invoke({'format_instructions': format_instructions})
```


```python
# loop through response as it is list
for r in response:
  print(r)
```

```
1. Autonomous vehicles for detecting pedestrians cyclists and other vehicles on the road
2. Retail stores for tracking inventory levels and monitoring product placement
3. Security systems for identifying unauthorized individuals entering restricted areas
4. Healthcare for analyzing medical images and detecting abnormalities or diseases
5. Agriculture for monitoring crop health and identifying pests or diseases in plants
```
