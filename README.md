# IncarnaMind

## 👀 In a Nutshell

IncarnaMind enables you to chat with your personal documents 📁 (PDF, TXT) using Language Learning Models (LLMs) like GPT (architecture overview). Our Sliding Window Chunking mechanism and Emsemble Retriever allow you to quickly find and search specific sections and both fine-grained and coarse-grained information within your documents to agument LLMs.

Powered by Langchain.

## 💡 Challenges Addressed

- **Fixed Chunking**: Our Sliding Window Chunking technique provides a balanced solution in terms of time, computing power, and performance.

- **Precision vs. Semantics**: Small chunks enable fine-grained information retrieval, while large chunks focus on coarse-grained data. We leverage both embedding-based and BM25 methods for a hybrid search approach.

- **Single-Document Limitation**: IncarnaMind supports multi-document querying, breaking the one-document-at-a-time barrier.

- **Model Compatibility**: We use Chains instead of Agent to ensure stable parsing across different LLMs.

## 🎯 Key Features

- **Sliding Window Chunking**: Dynamically adjust the size and position of text chunks to improve retrieval augmented generation (RAG).

- **Multi-Document QA**: Perform queries across multiple documents simultaneously.

- **File Compatibility**: Supports both PDF and TXT file formats.

- **LLM Model Compatibility**: Supports both OpenAI GPT and Anthropic Claude models.
  
## 💻 Demo

A short video

## 🚀 Getting Started

### 1. Installation

The installation is simple, you just need run few commands.

#### 1.0. Prerequisites

- 3.8 ≤ Python < 3.11 with [Conda](https://www.anaconda.com/download)
- [OpenAI API Key](https://beta.openai.com/signup) or [Anthropic Claude API Key](https://console.anthropic.com/account/keys)
- And of course, your own documents.

#### 1.1. Clone the repository

```shell
git clone https://github.com/junruxiong/IncarnaMind
cd IncarnaMind
```

#### 1.2. Setup

Create Conda virtual environment

```shell
conda create -n IncarnaMind python=3.10
```

Activate

```shell
conda activate IncarnaMind
```

Install all requirements

```shell
pip install -r requirements.txt
```

Setup your API keys in **configparser.ini** file

```shell
***REMOVED***
***REMOVED***(replace_me)
and/or
***REMOVED***(replace_me)
```

(Optional) Setup your custom parameters in **configparser.ini** file

```shell
***REMOVED***
PARAMETERS 1 = (replace_me)
PARAMETERS 2 = (replace_me)
...
PARAMETERS n = (replace_me)
```

### 2. Usage

#### 2.1. Upload and process your files

Put all your files (please name each file correctly to maximize the performance) into the **/data** directory and run the following command to ingest all data:

```shell
python docs2db.py
```

#### 2.2. Run

In order to start the conversation, run a command like:

```shell
python main.py
```

#### 2.3. Chat and ask any questions

Wait for the script to require your input like the below.

```shell
Human:
```

#### 2.4. Others

If you want to enable/disable the logging, please edit in in **configparser.ini** file.

```shell
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
```

## 👀 Architecture

### High Level Architecture

![image](figs/High_Level_Architecture.png)

### Sliding Window Chunking

![image](figs/Sliding_Window_Chunking.png)

## 🚫 Limitations

- Citation is not supported for now.
- Asynchronous is not very good for now.

## 📝 Upcoming Features

- UI
- OCR
- Optimise Async call
- Support Open Source LLMs
- support more document formats

## 📑 License

Apache 2.0 License
