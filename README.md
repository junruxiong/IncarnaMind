# 🧠 IncarnaMind

## 👀 In a Nutshell

IncarnaMind enables you to chat with your personal documents 📁 (PDF, TXT) using Large Language Models (LLMs) like GPT ([architecture overview](#high-level-architecture)). While OpenAI has recently launched a fine-tuning API for GPT models, it doesn't enable the base pretrained models to learn new data, and the responses can be prone to factual hallucinations. Utilize our [Sliding Window Chunking](#sliding-window-chunking) mechanism and Ensemble Retriever enables efficient querying of both fine-grained and coarse-grained information within your ground truth documents to augment the LLMs.

Feel free to use it and we welcome any feedback and new feature suggestions 🙌.

## ✨ New Updates

### Open-Source and Local LLMs Support

- **Recommended Model:** We've primarily tested with the Llama2 series models and recommend using [llama2-70b-chat](https://huggingface.co/TheBloke/Llama-2-70B-chat-GGUF) (either full or GGUF version) for optimal performance. Feel free to experiment with other LLMs.
- **System Requirements:** It requires more than 35GB of GPU RAM to run the GGUF quantized version.

### Alternative Open-Source LLMs Options

- **Insufficient RAM:** If you're limited by GPU RAM, consider using the [Together.ai](https://api.together.xyz/playground) API. It supports llama2-70b-chat and most other open-source LLMs. Plus, you get $25 in free usage.
- **Upcoming:** Smaller and cost-effecitive, fine-tuned models will be released in the future.

### How to use GGUF models

- For instructions on acquiring and using quantized GGUF LLM (similar to GGML), please refer to this [video](https://www.youtube.com/watch?v=lbFmceo4D5E) (from 10:45 to 12:30)..

Here is a comparison table of the different models I tested, for reference only:

| Metrics   | GPT-4  | GPT-3.5 | Claude 2.0 | Llama2-70b | Llama2-70b-gguf | Llama2-70b-api |
|-----------|--------|---------|------------|------------|-----------------|----------------|
| Reasoning | High   | Medium  | High       | Medium     | Medium          | Medium         |
| Speed     | Medium | High    | Medium     | Very Low   | Low             | Medium         |
| GPU RAM   | N/A    | N/A     | N/A        | Very High  | High            | N/A            |
| Safety    | Low    | Low     | Low        | High       | High            | Low            |

## 💻 Demo

https://github.com/junruxiong/IncarnaMind/assets/44308338/89d479fb-de90-4f7c-b166-e54f7bc7344c

## 💡 Challenges Addressed

- **Fixed Chunking**: Traditional RAG tools rely on fixed chunk sizes, limiting their adaptability in handling varying data complexity and context.

- **Precision vs. Semantics**:  Current retrieval methods usually focus either on semantic understanding or precise retrieval, but rarely both.

- **Single-Document Limitation**: Many solutions can only query one document at a time, restricting multi-document information retrieval.

- **Stability**: IncarnaMind is compatible with OpenAI GPT, Anthropic Claude, Llama2, and other open-source LLMs, ensuring stable parsing.

## 🎯 Key Features

- **Adaptive Chunking**: Our Sliding Window Chunking technique dynamically adjusts window size and position for RAG, balancing fine-grained and coarse-grained data access based on data complexity and context.

- **Multi-Document Conversational QA**: Supports simple and multi-hop queries across multiple documents simultaneously, breaking the single-document limitation.

- **File Compatibility**: Supports both PDF and TXT file formats.

- **LLM Model Compatibility**: Supports OpenAI GPT, Anthropic Claude, Llama2 and other open-source LLMs.

## 🏗 Architecture

### High Level Architecture

![image](figs/High_Level_Architecture.png)

### Sliding Window Chunking

![image](figs/Sliding_Window_Chunking.png)

## 🚀 Getting Started

### 1. Installation

The installation is simple, you just need to run few commands.

#### 1.0. Prerequisites

- 3.8 ≤ Python < 3.11 with [Conda](https://www.anaconda.com/download)
- One/All of [OpenAI API Key](https://beta.openai.com/signup), [Anthropic Claude API Key](https://console.anthropic.com/account/keys), [Together.ai API KEY](https://api.together.xyz/settings/api-keys) or [HuggingFace toekn for Meta Llama models](https://huggingface.co/settings/tokens)
- And of course, your own documents.

#### 1.1. Clone the repository

```shell
git clone https://github.com/junruxiong/IncarnaMind
cd IncarnaMind
```

#### 1.2. Setup

Create Conda virtual environment:

```shell
conda create -n IncarnaMind python=3.10
```

Activate:

```shell
conda activate IncarnaMind
```

Install all requirements:

```shell
pip install -r requirements.txt
```

Install [llama-cpp](https://github.com/abetlen/llama-cpp-python) seperatly if you want to run quantized local LLMs:

- For `NVIDIA` GPUs support, use `cuBLAS`

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

- For Apple Metal (`M1/M2`) support, use

```shell
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

Setup your one/all of API keys in **configparser.ini** file:

```shell
[tokens]
OPENAI_API_KEY = (replace_me)
ANTHROPIC_API_KEY = (replace_me)
TOGETHER_API_KEY = (replace_me)
# if you use full Meta-Llama models, you may need Huggingface token to access.
HUGGINGFACE_TOKEN = (replace_me)
```

(Optional) Setup your custom parameters in **configparser.ini** file:

```shell
[parameters]
PARAMETERS 1 = (replace_me)
PARAMETERS 2 = (replace_me)
...
PARAMETERS n = (replace_me)
```

### 2. Usage

#### 2.1. Upload and process your files

Put all your files (please name each file correctly to maximize the performance) into the **/data** directory and run the following command to ingest all data:
(You can delete example files in the **/data** directory before running the command)

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

When you start a chat, the system will automatically generate a **IncarnaMind.log** file.
If you want to edit the logging, please edit in the **configparser.ini** file.

```shell
[logging]
enabled = True
level = INFO
filename = IncarnaMind.log
format = %(asctime)s [%(levelname)s] %(name)s: %(message)s
```

## 🚫 Limitations

- Citation is not supported for current version, but will release soon.
- Limited asynchronous capabilities.

## 📝 Upcoming Features

- Frontend UI interface
- Fine-tuned small size open-source LLMs
- OCR support
- Asynchronous optimization
- Support more document formats

## 🙌 Acknowledgements

Special thanks to [Langchain](https://github.com/langchain-ai/langchain), [Chroma DB](https://github.com/chroma-core/chroma), [LocalGPT](https://github.com/PromtEngineer/localGPT), [Llama-cpp](https://github.com/abetlen/llama-cpp-python) for their invaluable contributions to the open-source community. Their work has been instrumental in making the IncarnaMind project a reality.

## 🖋 Citation

If you want to cite our work, please use the following bibtex entry:

```bibtex
@misc{IncarnaMind2023,
  author = {Junru Xiong},
  title = {IncarnaMind},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/junruxiong/IncarnaMind}}
}
```

## 📑 License

[Apache 2.0 License](LICENSE)