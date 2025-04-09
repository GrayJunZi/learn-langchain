# Venation

使用 Langchain 与 Ollama 和 本地LLM 构建和测试 AI Agent / RAG / ChatBot

# 一、介绍
## 什么是 LangChain？
LangChain 是一个用于开发由大型语言模型(Large Language Models, **LLMs**)驱动的应用程序的框架。

## LangChain 能做什么？
+ Chatbots
    - Chatbots 是聊天机器人，例如：ChatGPT、问答系统、客户支持应用。
+ RAG
    - 利用文档和知识源等外部数据源扩展LLMs知识。
+ AI Agents
    - AI Agents 可以计划、推理和执行行动。
+ AI 强大搜索引擎
+ 文本总结与生成
+ 数据提取与处理

## 为什么使用LangChain
### 标准化组件接口
人工智能应用模型和相关组件的数量不断增加，导致开发者需要学习和使用的API种类繁多。这种多样性可能会让开发者在构建应用时难以在不同提供商之间切换或组合组件。LangChain为关键组件提供了一个标准接口，使得在不同提供商之间切换变得容易

### 可观察性和评估
随着应用程序变得越来越复杂，越来越难以理解它们内部正在发生的事情。

## LangChain生态系统
### 什么是 LangSmith
LangSmith是由LangChain构建的LLM应用程序的评估和调试平台。

它通过提供强大的工具来跟踪、记录、调试和基准测试不同的LLM组件，帮助开发人员测试、优化和监控他们的AI应用程序。

跟踪和评估您的语言模型应用程序和智能代理，以帮助您从原型过渡到生产。

### LangGraph
LangGraph是一个用于构建具有LLMs的状态和多主体应用程序的库，用于创建代理和多代理工作流。

# 二、Ollama
## 什么是Ollama？
Ollama可以作为构建基于语言模型的应用程序的媒介，与本地模型一起使用LangChain。

## 下载 Ollama
进入官网进行下载 [Download Ollama on Windows](https://ollama.com/download)

[GitHub - ollama/ollama: Get up and running with Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3, and other large language models.](https://github.com/ollama/ollama)

或者进入Github官方仓库进行下载。

## 模型的选择
当选择要在本地运行的模型时，通常要选择型号：`7b`、`8b`、`14b`、`32b`等，其中的`b`代表的是十亿参数，也就是 70亿参数、80亿参数等、140亿和320亿参数等。目前 `deepseek-r1`模型中最大的参数为 6710亿参数。

## Ollama 命令
使用 `ollama list`来列出所有本地的模型。

```bash
ollama list
```

使用 `ollama run`下载并运行指定的模型。

```bash
ollama run deepseek-r1:14b
```

使用 `ollama rm`删除模型

```bash
ollama rm deepseek-r1:14b
```

## GUI 可视化界面
我们可以使用以下可视化工具，将我们本地的模型运行在上面。

[Msty - Using AI Models made Simple and Easy](https://msty.app/)

[GPT4All – The Leading Private AI Chatbot for Local Language Models](https://www.nomic.ai/gpt4all)

## Ollama API
我们可以使用 `ollama serve`命令来运行服务，默认监听的端口号是 `11434`。

如果服务已经启动了，那么将会输出以下报错信息。

![](https://cdn.nlark.com/yuque/0/2025/png/516447/1743832045801-f41a77fc-1b22-4255-a2c8-2c1cddfd97b5.png)

ollama提供了聊天接口`api/generate`通过以下参数来进行调用。

```bash
{
  "model": "deepseek-r1",
  "prompt": "为什么天空是蓝色的？"
}
```

默认输出的结果是流式的，可以在参数中增加 `stream`并设置为 `false`，那么它将会一次性输出结果。

# 三、与LLM进行交互

## 安装环境

创建Python虚拟环境

```bash
python3.12 -m venv learn-langchain_env
```

在`Linux/MacOS`下使用以下命令激活虚拟环境。

```bash
source learn-langchain_env/bin/activate
```

在 Windows环境下直接运行运行`activate`即可。
```bash
.\learn-langchain_env\Scripts\activate
```

## 使用 Jupyter Notebook
### 安装Jupyter Notebook
在 `Visual Studio Code`中安装 `Jupyter Notebook`扩展插件。

## 修改 pip 镜像源
运行以下命令：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 安装依赖
```bash
!pip install langchain
!pip install langchain-ollama
!pip install python-dotenv
```

## 加载 .env 文件
```python
from dotenv import load_dotenv

import os

load = load_dotenv('./../.env')

print(os.getenv("LANGSMITH_API_KEY"))
```

## 与LLM进行交互
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    temperature=0.5,
    max_tokens=250
)

response = llm.invoke("你好，你今天怎么样？")

print(response)
```

## 提示词与聊天模板
```python
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("What is the advantage of running the LLM in {env}")

prompt = prompt_template.invoke({"env":"local machine"})

llm.invoke(prompt)
```

### 提示词模板
```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

systemMessage = SystemMessagePromptTemplate.from_template("You are an LLM expert")
humanMessage = HumanMessagePromptTemplate.from_template("What is the advantage of running AI Models in {env}")

prompt_template = ChatPromptTemplate([
    systemMessage,
    humanMessage
])

prompt = prompt_template.invoke({"env":"local machine"})

content = llm.invoke(prompt)

print(content)
```

## 消息占位
```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate([
    ("system", "You are an LLM expert"),
    MessagesPlaceholder("msg")
])

prompt = prompt_template.invoke({"msg":[HumanMessage("What is the advantage of running LLM in local machine")]})

for str in llm.stream(prompt):
    print(str.content)
```

