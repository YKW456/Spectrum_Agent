# Spectrum Agent: Natural Language-Driven Optical Multi-layer Film Structure Designer

Spectrum Agent is a natural language-driven intelligent agent for designing optical multilayer film structures. It is fine-tuned based on domain knowledge extracted from a large number of research papers and tool invocation examples. It can recommend materials for structural design based on natural language input and invoke the PPO-BO algorithm to carry out the structure design.
The literature used in this study was downloaded from the Internet. During the paper review phase, all code and datasets were made available on an anonymous GitHub repository. After the paper is published, we plan to provide all the code used in this study, as well as datasets free of copyright issues.
## Environment Setup
- Install python packages:
```bash
pip install -r requirements.txt
```
- Download llms:

Download [DeepSeek7b](https://www.modelscope.cn/models/deepseek-ai/deepseek-llm-7b-base/feedback) to: SpectrumAgent/Model/Gemma2_9b

Download [Qwen2.5 7b](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B) to: SpectrumAgent/Qwen2.5_7b

Download [Mistral 7b](https://www.modelscope.cn/models/AI-ModelScope/Mistral-7B-v0.1/summary) to: SpectrumAgent/Model/Mistral_7b

Download [LLama3.1 8b](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B/files)  to: SpectrumAgent/Model/Llama3.1_8b

Download [Gemma2 9b](https://www.modelscope.cn/models/LLM-Research/gemma-2-9b-it/summary) to: SpectrumAgent/Model/Gemma2_9b

- Download a sentence transformer model:
  
Download [Sentence transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
to: SpectrumAgent/Model/allMiniLM

## Using llm to recommend materials for your design. 
If you want to finetune LLMs to improve the quality of professional knowledge responses, you can run:
```bash
cd SFT-knowledge
python lora_sft_main.py
```
If you want to finetune LLMs to enhance its tool invocation capabilities, you can run:
```bash
cd SFT-tool
python lora_main.py
```
If you want to evaluate the response quality of different large language models, you can run:

```bash
cd Assessment-Knowledge
python Assessment.py
```
If you want to evaluate the tool invocation accuracy of different large language models, you can run:

```bash
cd Assessment-Tool
python Assessment.py
```
If you want to implement material recommendation without a local LLM (the sentence transformer model must be downloaded, but there's no need to download the LLMs, though an API key is required), and automatically configure and adjust the PPO-BO algorithm for structural design, you can configure the API key, and then:
```bash
python chat_and_design.py
```

An important feature of this repository is that it allows users to automatically design structures based on natural language. However, users can still manually modify the configuration file to adjust the search objectives of the PPO-BO algorithm, such as the materials used, the search range, and other related information. The configuration of the PPO-BO algorithm depends on the bayes_config.py file, which you can modify, and then:
```bash
python run_single_PPO_BO_optimization.py
```
