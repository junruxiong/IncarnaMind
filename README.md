# IncarnaMind

## 👀 In a Nutshell

Chat with your personal documents.
Upload documents 📁(PDF, TXT) and answer questions about them.

together gif:
Sliding Window Chunking (diagram)
Whole workflow (diagram)

![image](figs/Sliding_Window_Chunking.png)

## 💢 Problems

-current chunking is too fixed
sliding windows chunking, we tried other methods, this one is balanced with time, computing power and performance.

-small chunks can retrieve fine grained information, large chunks can retrieve coarse grained information, embedding based retrieval are good at semantic searching, howerver like keywords searching and bm25 are good at precision matching, we emsemble both of them to retrieve information.

-Normally, only qa with 1 document each time
multiple documents qa

## 🎯 Key Features

- **Sliding window chunking**
- **Multi-docs QA**
- **File compatibility**
  - .pdf
  - .txt

## 💻 Demo

A short video

## 🚀 Getting Started

### 1. Installation

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

Put all your files into the **/data** directory and run the following command to ingest all the data:

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

Enable/disable logging
Please don't use random name of each files

## 🚫 Limitations

- Asynchronous is not very good for now.

## 📝 Upcoming Features

- UI
- OCR
- Async call
- Support Open Source LLMs
- support more formats

## 📑 License
