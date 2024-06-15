# 🧠 IncarnaMind

## 👀 In a Nutshell

IncarnaMind enables you to chat with your personal documents 📁 (PDF, TXT) using Large Language Models (LLMs) like GPT ([architecture overview](#high-level-architecture)). While OpenAI has recently launched a fine-tuning API for GPT models, it doesn't enable the base pretrained models to learn new data, and the responses can be prone to factual hallucinations. Utilize our [Sliding Window Chunking](#sliding-window-chunking) mechanism and Ensemble Retriever enables efficient querying of both fine-grained and coarse-grained information within your ground truth documents to augment the LLMs.

## 1.2. Setup

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

- 
## 📝 Upcoming Features

- 

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