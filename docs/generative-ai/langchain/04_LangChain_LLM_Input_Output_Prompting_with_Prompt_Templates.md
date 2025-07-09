---
title: 'Prompting with Prompt Templates'
date: 2025-04-05

parent: Generative AI with LangChain

nav_order: 4

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Prompting with Prompt Templates
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

Large Language Models (LLMs) are essential to LangChain, which does not create its own LLMs but offers a standardized API to interact with various LLM providers like OpenAI and Hugging Face. The LLM class in LangChain ensures a consistent interface for engaging with these diverse LLMs.

<!-- **Enter API Token**


```python
import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
``` -->

**LLM ChatOpenAI**


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
            model_name = "gpt-3.5-turbo",
            temperature=0
        )
```

## Prompt Templates

Prompt templates are pre-designed formats used to generate prompts for language models. These templates can include instructions, examples, and specific contexts and questions suited for particular tasks.

**Types of Prompt Templates:**

*   **PromptTemplate**: Utilized for creating string-based prompts, supporting Python's `str.format` syntax for templating. It accommodates any number of variables, including scenarios without variables.

*   **ChatPromptTemplate**: Specifically designed for chat models, where the prompt is structured as a list of chat messages. Each message includes content and a role parameter, distinguishing between AI assistant, human, or system roles.

*   **FewShotChatMessagePromptTemplate**: This template is created from a set of examples

### PromptTemplate

We utilize `PromptTemplate` to create structured string prompts, utilizing Python's `str.format` syntax by default for templating. This allows customization of prompt formats according to specific needs. For more details, refer [Prompt Template Composition](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/composition/).

**Single Prompt**


```python
from langchain.prompts import PromptTemplate

prompt = """Explain what is Machine Learning in 3 points"""
prompt_template = PromptTemplate.from_template(prompt)
print(prompt_template)
```

```
PromptTemplate(input_variables=[], template='Explain what is Machine Learning in 3 points')
```


```python
prompt_template.format()
```
```
'Explain what is Machine Learning in 3 points'
```



```python
response = llm.invoke(prompt_template.format())
print(response.content)
```
```
1. Machine learning is a subset of artificial intelligence that involves the development of algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data without being explicitly programmed.

2. Machine learning algorithms use patterns and insights from data to improve their performance over time, allowing them to adapt and make more accurate predictions or decisions as they are exposed to more data.

3. Machine learning is used in a wide range of applications, including image and speech recognition, natural language processing, recommendation systems, and autonomous vehicles, among others. It has the potential to revolutionize industries and improve efficiency and accuracy in various tasks.
```

**More complex Prompt with Placeholders**


```python
prompt = """Explain briefly about {subject} in {language}."""

prompt_template = PromptTemplate.from_template(prompt)
print(prompt_template)
```

```
PromptTemplate(input_variables=['language', 'subject'], template='Explain briefly about {subject} in {language}.')
```



```python
inputs = [
      ("Machine Learning", "English"),
      ("Deep Learning", "Spanish"),
      ("Artificial Intelligence", "Hindi")
  ]

prompts = [
    prompt_template.format(subject=subject, language=language) for subject, language in inputs
]
print(prompts)
```

```
['Explain briefly about Machine Learning in English.',
'Explain briefly about Deep Learning in Spanish.',
'Explain briefly about Artificial Intelligence in Hindi.']
```


```python
# use map function to run on multiple prompts in one go
responses = llm.map().invoke(prompts)
print(responses)
```



```
[AIMessage(content='Machine learning is a type of artificial intelligence that allows computers to learn and improve from experience without being explicitly programmed. It involves algorithms that analyze and interpret data to make predictions or decisions. Machine learning is used in a wide range of applications, such as image and speech recognition, recommendation systems, and autonomous vehicles.', response_metadata={'token_usage': {'completion_tokens': 61, 'prompt_tokens': 16, 'total_tokens': 77}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}),
AIMessage(content='El aprendizaje profundo, también conocido como deep learning, es una rama del aprendizaje automático que se basa en redes neuronales artificiales para imitar el funcionamiento del cerebro humano y aprender a partir de grandes cantidades de datos. Estas redes neuronales tienen múltiples capas de procesamiento que les permiten extraer características complejas de los datos y realizar tareas como reconocimiento de voz, visión por computadora, procesamiento del lenguaje natural, entre otros. El aprendizaje profundo ha demostrado ser muy efectivo en una amplia variedad de aplicaciones y está en constante evolución con nuevos avances y desarrollos en el campo de la inteligencia artificial.', response_metadata={'token_usage': {'completion_tokens': 156, 'prompt_tokens': 16, 'total_tokens': 172}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}),
AIMessage(content='कृत्रिम बुद्धिमत्ता एक शाखा है जो कंप्यूटर और मशीनों को मानव बुद्धिमत्ता की तरह काम करने की क्षमता प्रदान करती है। यह तकनीकी उन्नति का एक महत्वपूर्ण क्षेत्र है जिसमें कंप्यूटर और रोबोट को आत्म-संज्ञान, सोचने, सीखने और समस्याओं का समाधान करने की क्षमता प्राप्त होती है। इसका उद्देश्य है कि मशीनें मानवों की सहायता करें और उनके लिए कार्य सरल और आसान बनाएं।', response_metadata={'token_usage': {'completion_tokens': 338, 'prompt_tokens': 16, 'total_tokens': 354}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None})]
```



```python
for response in responses:
  print(response.content)
  print()
```

```
Machine learning is a type of artificial intelligence that allows computers to learn and improve from experience without being explicitly programmed. It involves algorithms that analyze and interpret data to make predictions or decisions. Machine learning is used in a wide range of applications, such as image and speech recognition, recommendation systems, and autonomous vehicles.

El aprendizaje profundo, también conocido como deep learning, es una rama del aprendizaje automático que se basa en redes neuronales artificiales para imitar el funcionamiento del cerebro humano y aprender a partir de grandes cantidades de datos. Estas redes neuronales tienen múltiples capas de procesamiento que les permiten extraer características complejas de los datos y realizar tareas como reconocimiento de voz, visión por computadora, procesamiento del lenguaje natural, entre otros. El aprendizaje profundo ha demostrado ser muy efectivo en una amplia variedad de aplicaciones y está en constante evolución con nuevos avances y desarrollos en el campo de la inteligencia artificial.

कृत्रिम बुद्धिमत्ता एक शाखा है जो कंप्यूटर और मशीनों को मानव बुद्धिमत्ता की तरह काम करने की क्षमता प्रदान करती है। यह तकनीकी उन्नति का एक महत्वपूर्ण क्षेत्र है जिसमें कंप्यूटर और रोबोट को आत्म-संज्ञान, सोचने, सीखने और समस्याओं का समाधान करने की क्षमता प्राप्त होती है। इसका उद्देश्य है कि मशीनें मानवों की सहायता करें और उनके लिए कार्य सरल और आसान बनाएं।
```


### ChatPromptTemplate


The standard prompt format for chat models involves a list of chat messages. Each message contains content and an additional parameter called `role`.

For instance, in the OpenAI Chat Completions API, a chat message can be assigned roles such as AI assistant, human, or system.

**simple prompt with placeholder**


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = """Explain briefly about {topic}."""
chat_template = ChatPromptTemplate.from_template(prompt)
print(chat_template)
```

```
ChatPromptTemplate(input_variables=['topic'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['topic'], template='Explain briefly about {topic}.'))])
```



```python
topics = ['Dividend in stock market', 'Equity income', 'Profit margin']
prompts = [chat_template.format(topic = topic) for topic in topics]
print(prompts)
```


```
['Human: Explain briefly about Dividend in stock market.',
'Human: Explain briefly about Equity income.',
'Human: Explain briefly about Profit margin.']
```

```python
responses = llm.map().invoke(prompts)
print(responses)
```

```
[AIMessage(content="Dividend is a portion of a company's profits that is distributed to its shareholders. It is usually paid out on a regular basis, such as quarterly or annually, and is typically in the form of cash or additional shares of stock. Dividends are a way for companies to reward their shareholders for investing in the company and can provide a steady income stream for investors. The amount of dividend paid out is determined by the company's board of directors and can vary based on the company's financial performance and overall strategy.", response_metadata={'token_usage': {'completion_tokens': 103, 'prompt_tokens': 18, 'total_tokens': 121}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}),
AIMessage(content="Equity income refers to the portion of a company's profit that is distributed to shareholders in the form of dividends. This income is derived from the company's ownership of stocks or equity investments in other companies. It is a source of passive income for investors and can provide a steady stream of cash flow. Equity income can be a key component of a diversified investment portfolio, providing both income and potential for capital appreciation.", response_metadata={'token_usage': {'completion_tokens': 82, 'prompt_tokens': 15, 'total_tokens': 97}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}),
AIMessage(content="Profit margin is a financial metric that measures the profitability of a company by calculating the percentage of revenue that exceeds the company's costs. It is calculated by dividing the company's net income by its total revenue and multiplying the result by 100 to get a percentage. A higher profit margin indicates that a company is more efficient at generating profits from its revenue.", response_metadata={'token_usage': {'completion_tokens': 70, 'prompt_tokens': 15, 'total_tokens': 85}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None})]
```

```python
for response in responses:
  print(response.content)
  print()
```

```
Dividend is a portion of a company's profits that is distributed to its shareholders. It is usually paid out on a regular basis, such as quarterly or annually, and is typically in the form of cash or additional shares of stock. Dividends are a way for companies to reward their shareholders for investing in the company and can provide a steady income stream for investors. The amount of dividend paid out is determined by the company's board of directors and can vary based on the company's financial performance and overall strategy.

Equity income refers to the portion of a company's profit that is distributed to shareholders in the form of dividends. This income is derived from the company's ownership of stocks or equity investments in other companies. It is a source of passive income for investors and can provide a steady stream of cash flow. Equity income can be a key component of a diversified investment portfolio, providing both income and potential for capital appreciation.

Profit margin is a financial metric that measures the profitability of a company by calculating the percentage of revenue that exceeds the company's costs. It is calculated by dividing the company's net income by its total revenue and multiplying the result by 100 to get a percentage. A higher profit margin indicates that a company is more efficient at generating profits from its revenue.
```

```python
print(responses[0])
```



```
AIMessage(content="Dividend is a portion of a company's profits that is distributed to its shareholders. It is usually paid out on a regular basis, such as quarterly or annually, and is typically in the form of cash or additional shares of stock. Dividends are a way for companies to reward their shareholders for investing in the company and can provide a steady income stream for investors. The amount of dividend paid out is determined by the company's board of directors and can vary based on the company's financial performance and overall strategy.", response_metadata={'token_usage': {'completion_tokens': 103, 'prompt_tokens': 18, 'total_tokens': 121}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None})
```


**More complex prompt with series of messages**


```python
messages = [
        ("system", "Act as an expert in cloud computing and provide brief answers"),
        ("human", "What is your name?"),
        ("ai", "My name is AIbot"),
        ("human", "{user_prompt}")
    ]

chat_template = ChatPromptTemplate.from_messages(messages)
print(chat_template)
```
```
ChatPromptTemplate(input_variables=['user_prompt'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='Act as an expert in cloud computing and provide brief answers')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='What is your name?')), AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='My name is AIbot')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['user_prompt'], template='{user_prompt}'))])
```



```python
text_prompts = [
        "what is your name?",
        "Can you explain to me what is cloud computing?"
      ]

chat_prompts = [chat_template.format(user_prompt = prompt) for prompt in text_prompts]
print(chat_prompts)
```

```
['System: Act as an expert in cloud computing and provide brief answers\nHuman: What is your name?\nAI: My name is AIbot\nHuman: what is your name?',
'System: Act as an expert in cloud computing and provide brief answers\nHuman: What is your name?\nAI: My name is AIbot\nHuman: Can you explain to me what is cloud computing?']
```


```python
print(chat_prompts[0])
```

```
System: Act as an expert in cloud computing and provide brief answers
Human: What is your name?
AI: My name is AIbot
Human: what is your name?
```


```python
print(chat_prompts[1])
```

```
System: Act as an expert in cloud computing and provide brief answers
Human: What is your name?
AI: My name is AIbot
Human: Can you explain to me what is cloud computing?
```


```python
responses = llm.map().invoke(chat_prompts)

for response in responses:
  print(response.content)
  print()
```

```
AI: My name is AIbot

AI: Cloud computing is the delivery of computing services such as servers, storage, databases, networking, software, and analytics over the internet (the cloud) to offer faster innovation, flexible resources, and economies of scale.
```



```python
messages = [
        ("system", "Act as an expert in cloud computing and provide very detailed answers with suitable examples"),
        ("human", "What is your name?"),
        ("ai", "My name is AIbot"),
        ("human", "{user_prompt}")
    ]

chat_template = ChatPromptTemplate.from_messages(messages)
print(chat_template)
```

```
ChatPromptTemplate(input_variables=['user_prompt'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='Act as an expert in cloud computing and provide very detailed answers with suitable examples')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='What is your name?')), AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='My name is AIbot')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['user_prompt'], template='{user_prompt}'))])
```



```python
text_prompts = [
        "what is your name?",
        "Can you explain to me what is cloud computing?"
      ]

chat_prompts = [chat_template.format(user_prompt = prompt) for prompt in text_prompts]
print(chat_prompts)
```


```
['System: Act as an expert in cloud computing and provide very detailed answers with suitable examples\nHuman: What is your name?\nAI: My name is AIbot\nHuman: what is your name?',
'System: Act as an expert in cloud computing and provide very detailed answers with suitable examples\nHuman: What is your name?\nAI: My name is AIbot\nHuman: Can you explain to me what is cloud computing?']
```



```python
responses = llm.map().invoke(chat_prompts)

for response in responses:
  print(response.content)
  print("--------------------------------------------------")
```

```
AI: My name is AIbot.
--------------------------------------------------
AI: Cloud computing is the delivery of computing services over the internet, allowing users to access and store data, run applications, and utilize resources on remote servers rather than on their local devices. This technology enables users to access their data and applications from anywhere with an internet connection, providing flexibility, scalability, and cost-effectiveness.

There are three main types of cloud computing services: Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). 

- IaaS provides virtualized computing resources over the internet, such as virtual machines, storage, and networking. Examples of IaaS providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform.

- PaaS offers a platform for developers to build, deploy, and manage applications without having to worry about the underlying infrastructure. Examples of PaaS providers include Heroku, Google App Engine, and Microsoft Azure App Service.

- SaaS delivers software applications over the internet on a subscription basis, eliminating the need for users to install and maintain software on their devices. Examples of SaaS applications include Google Workspace, Microsoft 365, and Salesforce.

Overall, cloud computing provides businesses and individuals with the ability to access powerful computing resources on-demand, without the need for significant upfront investments in hardware and infrastructure.
--------------------------------------------------
```

{: .note }
PromptTemplate and ChatPromptTemplate supports LangChain Expression Language (LCEL).


```python
text_prompts = [
        "what is your name?",
        "Can you explain to me what is cloud computing?"
      ]

chat_prompts = [chat_template.invoke({"user_prompt" : prompt}) for prompt in text_prompts]
print(chat_prompts)
```

```
[ChatPromptValue(messages=[SystemMessage(content='Act as an expert in cloud computing and provide very detailed answers with suitable examples'), HumanMessage(content='What is your name?'), AIMessage(content='My name is AIbot'), HumanMessage(content='what is your name?')]),
ChatPromptValue(messages=[SystemMessage(content='Act as an expert in cloud computing and provide very detailed answers with suitable examples'), HumanMessage(content='What is your name?'), AIMessage(content='My name is AIbot'), HumanMessage(content='Can you explain to me what is cloud computing?')])]
```


```python
print(chat_prompts[1])
```

```
ChatPromptValue(messages=[SystemMessage(content='Act as an expert in cloud computing and provide very detailed answers with suitable examples'), HumanMessage(content='What is your name?'), AIMessage(content='My name is AIbot'), HumanMessage(content='Can you explain to me what is cloud computing?')])
```



```python
print(chat_prompts[1].to_string())
```

```
System: Act as an expert in cloud computing and provide very detailed answers with suitable examples
Human: What is your name?
AI: My name is AIbot
Human: Can you explain to me what is cloud computing?
```


```python
chat_prompts[1].to_messages()
```

```
[SystemMessage(content='Act as an expert in cloud computing and provide very detailed answers with suitable examples'),
HumanMessage(content='What is your name?'),
AIMessage(content='My name is AIbot'),
HumanMessage(content='Can you explain to me what is cloud computing?')]
```



```python
responses = llm.map().invoke(chat_prompts)

for response in responses:
  print(response.content)
  print("--------------------------------------------------")
```

```
My name is AIbot
--------------------------------------------------
Cloud computing is the delivery of computing services over the internet, allowing users to access and use resources such as servers, storage, databases, networking, software, and analytics on a pay-as-you-go basis. Instead of owning and maintaining physical servers or infrastructure, users can leverage cloud service providers' resources to scale their operations quickly and efficiently.

There are three main types of cloud computing services:

1. Infrastructure as a Service (IaaS): Provides virtualized computing resources over the internet, such as virtual machines, storage, and networking. Users can deploy and manage their applications on these virtualized resources without worrying about the underlying infrastructure.

2. Platform as a Service (PaaS): Offers a platform for developers to build, deploy, and manage applications without having to worry about the underlying infrastructure. PaaS providers offer tools and services to streamline the development process.

3. Software as a Service (SaaS): Delivers software applications over the internet on a subscription basis. Users can access these applications through a web browser without needing to install or maintain the software locally.

Cloud computing offers several benefits, including scalability, flexibility, cost-effectiveness, and improved collaboration. For example, a company can quickly scale its infrastructure to handle increased demand during peak times without investing in additional hardware. Additionally, cloud services enable remote teams to collaborate effectively by providing access to shared resources from anywhere with an internet connection.
--------------------------------------------------
```

### FewShotChatMessagePromptTemplate

This Chat prompt template supports few-shot examples by structuring conversations into lists of messages. It includes prefix messages, example messages, and suffix messages.

This template allows for interactions like:

```
System: You are a helpful AI Assistant
Human: What is 3+3?
AI: 6
Human: What is 5*3?
AI: 15
Human: What is 4-1?
```

Define the few-shot examples you would like to include.


```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
```


```python
example_data = [
    ("""England and Spain have both secured their places in the Euro 2024 final, with standout performances from players like Ollie Watkins and Lamine Yamal.
    Meanwhile, India has qualified for the WCL semi-finals despite a loss to South Africa, and Lionel Messi continues to shine as Argentina reaches the Copa America final.""", 'Sports'),
    ("""A new study has revealed insights into Saturn's planet-wide storms, and India has announced plans to mandate USB-C as a common charger standard from June 2025​ (shortpedia)​.
    Additionally, scientists have developed self-healing polymers using waste materials, showing promising advancements in sustainable materials.""", 'Technology'),
    ("""The RBI is working with ASEAN to create a platform for fast cross-border retail payments, and SBI has raised ₹10,000 crore through 15-year infrastructure bonds.
    Furthermore, the International Toy Fair is set to begin with participation from over 100 foreign buyers​.""", 'Business'),
    ("""MIT researchers have developed a new air safety system called Air-Guardian, which functions as a proactive co-pilot to enhance safety during critical flight moments.
    Additionally, a new surgical procedure and neuroprosthetic interface from MIT allows individuals with amputations to control their prosthetic limbs with their brains, offering smoother gait and better navigation of obstacles​.""", 'Technology')
]

print(example_data)
```



```
[('England and Spain have both secured their places in the Euro 2024 final, with standout performances from players like Ollie Watkins and Lamine Yamal. \n    Meanwhile, India has qualified for the WCL semi-finals despite a loss to South Africa, and Lionel Messi continues to shine as Argentina reaches the Copa America final.',
'Sports'),
("A new study has revealed insights into Saturn's planet-wide storms, and India has announced plans to mandate USB-C as a common charger standard from June 2025\u200b (shortpedia)\u200b. \n    Additionally, scientists have developed self-healing polymers using waste materials, showing promising advancements in sustainable materials.",
'Technology'),
('The RBI is working with ASEAN to create a platform for fast cross-border retail payments, and SBI has raised ₹10,000 crore through 15-year infrastructure bonds.\n    Furthermore, the International Toy Fair is set to begin with participation from over 100 foreign buyers\u200b.',
'Business'),
('MIT researchers have developed a new air safety system called Air-Guardian, which functions as a proactive co-pilot to enhance safety during critical flight moments. \n    Additionally, a new surgical procedure and neuroprosthetic interface from MIT allows individuals with amputations to control their prosthetic limbs with their brains, offering smoother gait and better navigation of obstacles\u200b.',
'Technology')]
```


```python
example_data_formatted = [
    {
        "input": input,
        "output": output
    } for input, output in example_data
]
print(example_data_formatted)
```


```
[{'input': 'England and Spain have both secured their places in the Euro 2024 final, with standout performances from players like Ollie Watkins and Lamine Yamal. \n    Meanwhile, India has qualified for the WCL semi-finals despite a loss to South Africa, and Lionel Messi continues to shine as Argentina reaches the Copa America final.',
'output': 'Sports'},
{'input': "A new study has revealed insights into Saturn's planet-wide storms, and India has announced plans to mandate USB-C as a common charger standard from June 2025\u200b (shortpedia)\u200b. \n    Additionally, scientists have developed self-healing polymers using waste materials, showing promising advancements in sustainable materials.",
'output': 'Technology'},
{'input': 'The RBI is working with ASEAN to create a platform for fast cross-border retail payments, and SBI has raised ₹10,000 crore through 15-year infrastructure bonds.\n    Furthermore, the International Toy Fair is set to begin with participation from over 100 foreign buyers\u200b.',
'output': 'Business'},
{'input': 'MIT researchers have developed a new air safety system called Air-Guardian, which functions as a proactive co-pilot to enhance safety during critical flight moments. \n    Additionally, a new surgical procedure and neuroprosthetic interface from MIT allows individuals with amputations to control their prosthetic limbs with their brains, offering smoother gait and better navigation of obstacles\u200b.',
'output': 'Technology'}]
```



```python
# combine them into few-shot prompt template
data_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)
```

```python
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=data_prompt,
    examples=example_data_formatted
)

print(few_shot_prompt)
```



```
FewShotChatMessagePromptTemplate(examples=[{'input': 'England and Spain have both secured their places in the Euro 2024 final, with standout performances from players like Ollie Watkins and Lamine Yamal. \n    Meanwhile, India has qualified for the WCL semi-finals despite a loss to South Africa, and Lionel Messi continues to shine as Argentina reaches the Copa America final.', 'output': 'Sports'}, {'input': "A new study has revealed insights into Saturn's planet-wide storms, and India has announced plans to mandate USB-C as a common charger standard from June 2025\u200b (shortpedia)\u200b. \n    Additionally, scientists have developed self-healing polymers using waste materials, showing promising advancements in sustainable materials.", 'output': 'Technology'}, {'input': 'The RBI is working with ASEAN to create a platform for fast cross-border retail payments, and SBI has raised ₹10,000 crore through 15-year infrastructure bonds.\n    Furthermore, the International Toy Fair is set to begin with participation from over 100 foreign buyers\u200b.', 'output': 'Business'}, {'input': 'MIT researchers have developed a new air safety system called Air-Guardian, which functions as a proactive co-pilot to enhance safety during critical flight moments. \n    Additionally, a new surgical procedure and neuroprosthetic interface from MIT allows individuals with amputations to control their prosthetic limbs with their brains, offering smoother gait and better navigation of obstacles\u200b.', 'output': 'Technology'}], example_prompt=ChatPromptTemplate(input_variables=['input', 'output'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')), AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=['output'], template='{output}'))]))
```



```python
print(few_shot_prompt.format())
```

```
Human: England and Spain have both secured their places in the Euro 2024 final, with standout performances from players like Ollie Watkins and Lamine Yamal. Meanwhile, India has qualified for the WCL semi-finals despite a loss to South Africa, and Lionel Messi continues to shine as Argentina reaches the Copa America final.
AI: Sports
Human: A new study has revealed insights into Saturn's planet-wide storms, and India has announced plans to mandate USB-C as a common charger standard from June 2025​ (shortpedia)​. Additionally, scientists have developed self-healing polymers using waste materials, showing promising advancements in sustainable materials.
AI: Technology
Human: The RBI is working with ASEAN to create a platform for fast cross-border retail payments, and SBI has raised ₹10,000 crore through 15-year infrastructure bonds.Furthermore, the International Toy Fair is set to begin with participation from over 100 foreign buyers​.
AI: Business
Human: MIT researchers have developed a new air safety system called Air-Guardian, which functions as a proactive co-pilot to enhance safety during critical flight moments. Additionally, a new surgical procedure and neuroprosthetic interface from MIT allows individuals with amputations to control their prosthetic limbs with their brains, offering smoother gait and better navigation of obstacles​.
AI: Technology
```



```python
# Now, add final prompt and provide it to LLM
final_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify each news article using the following format"),
        few_shot_prompt,
        ("human", "{input}")
    ]
)
print(final_prompt_template)
```


```
ChatPromptTemplate(input_variables=['input'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='Classify each news article using the following format')), FewShotChatMessagePromptTemplate(examples=[{'input': 'England and Spain have both secured their places in the Euro 2024 final, with standout performances from players like Ollie Watkins and Lamine Yamal. \n    Meanwhile, India has qualified for the WCL semi-finals despite a loss to South Africa, and Lionel Messi continues to shine as Argentina reaches the Copa America final.', 'output': 'Sports'}, {'input': "A new study has revealed insights into Saturn's planet-wide storms, and India has announced plans to mandate USB-C as a common charger standard from June 2025\u200b (shortpedia)\u200b. \n    Additionally, scientists have developed self-healing polymers using waste materials, showing promising advancements in sustainable materials.", 'output': 'Technology'}, {'input': 'The RBI is working with ASEAN to create a platform for fast cross-border retail payments, and SBI has raised ₹10,000 crore through 15-year infrastructure bonds.\n    Furthermore, the International Toy Fair is set to begin with participation from over 100 foreign buyers\u200b.', 'output': 'Business'}, {'input': 'MIT researchers have developed a new air safety system called Air-Guardian, which functions as a proactive co-pilot to enhance safety during critical flight moments. \n    Additionally, a new surgical procedure and neuroprosthetic interface from MIT allows individuals with amputations to control their prosthetic limbs with their brains, offering smoother gait and better navigation of obstacles\u200b.', 'output': 'Technology'}], example_prompt=ChatPromptTemplate(input_variables=['input', 'output'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')), AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=['output'], template='{output}'))])), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))])
```



```python
docs = [
    """
    Researchers at POSTECH have developed a revolutionary method for synthesizing single-crystal cathode materials, significantly boosting the lifespan and efficiency of electric vehicle batteries.
    This breakthrough could extend the driving range and durability of EVs, making them more practical for everyday use. Additionally, scientists have created a 2D device that efficiently cools quantum systems, overcoming a significant hurdle in quantum computing development.
    """,
    """
    In Dubai, the year 2024 is packed with various business events such as the Dubai International Project Management Forum from January 15-18, focusing on sustainability and modern project management techniques.
    Another key event is the Intersec 2024 exhibition, held from January 16-18, which will showcase innovations in safety, security, and fire protection technology with over 1,000 exhibitors and 45,000 attendees expected.
    """
]
```


```python
final_prompts = [final_prompt_template.format(input=doc) for doc in docs]
```


```python
responses = llm.map().invoke(final_prompts)
print(responses)
```

```
[AIMessage(content='AI: Technology', response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 385, 'total_tokens': 388}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-f5753df3-7b2f-4b03-9153-40e40f68db48-0'),
AIMessage(content='AI: Business', response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 395, 'total_tokens': 398}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-bc819a78-65f3-4a18-abbf-0d010e479121-0')]
```



```python
for response in responses:
  print(response.content)
  print()
```

```
AI: Technology

AI: Business
```


### Partial Prompt Templates


When using prompt templates, like other methods, it can be beneficial to `partial` a template. This involves providing a subset of the required data values initially, creating a new template that expects the remaining data values later.


```python
from datetime import datetime

def _get_datetime():
  now = datetime.now()
  return now.strftime("%m/%d/%Y")
```


```python
prompt_txt = """write poem on {subject} on the date {date}"""
prompt = ChatPromptTemplate.from_template(prompt_txt)
```


```python
prompt = prompt.partial(date=_get_datetime)
print(prompt)
```


```
ChatPromptTemplate(input_variables=['subject'], partial_variables={'date': <function _get_datetime at 0x7e731a3c00d0>}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['date', 'subject'], template=`'write poem on {subject} on the date {date}'))])
```



```python
subjects = ["Science & Technology", "Physics"]
final_prompts = [prompt.format(subject = subject) for subject in subjects]
print(final_prompts)
```

```
['Human: write poem on Science & Technology on the date 07/13/2024',
'Human: write poem on Physics on the date 07/13/2024']
```


```python
responses = llm.map().invoke(final_prompts)
for response in responses:
  print(response.content)
  print("--------------------------------------------------")
```

```
In the realm of Science and Technology,
We unlock the secrets of the universe with curiosity.
From the smallest atom to the vast expanse of space,
We push the boundaries of knowledge at a rapid pace.

In labs and workshops, minds collaborate,
To innovate and create, to elucidate.
From AI to biotech, we strive to enhance,
The quality of life, to give humanity a chance.

With every discovery, we gain insight,
Into the mysteries of existence, shining bright.
We harness the power of innovation and invention,
To shape a future of progress and ascension.

So let us celebrate the wonders of Science and Tech,
For they hold the key to a world that's dynamic and eclectic.
May we continue to explore, to question, to dream,
And unlock the potential of this ever-evolving stream.
--------------------------------------------------
In the realm of atoms and particles unseen,
Physics reveals the secrets of the universe's machine.
From the laws of motion to the theory of relativity,
It unravels the mysteries with clarity and creativity.

Einstein's equations and Newton's laws,
Guide us through the cosmos with a scientific cause.
From the smallest quark to the largest star,
Physics explains it all, near and far.

Through experiments and calculations,
We explore the depths of space and time's fluctuations.
From black holes to quantum entanglement,
Physics pushes the boundaries of human engagement.

So let us marvel at the wonders of Physics,
As we delve into the unknown with curiosity and ethics.
For in the study of matter and energy's interaction,
We find the beauty and complexity of the universe's attraction.
--------------------------------------------------
```
