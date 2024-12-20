# README.md

## How do Multimodal Large Language Models (MLLMs) perform when applied to graphical perception tasks?

### 🟢 How to answer this question?

To answer this research question, our study focuses on evaluating MLLMs' performance on graphical perception, inspired by two popular studies from Cleveland and McGrill, and Haehn et al.

We incorporate our pretrained and fine-tuned MLLMs into our experiments. In particular, for fine-tuned models, we compress them by training on top of pretrained models, dig deep into their performance using efficient training configurations, scalable adaptation techniques to improve efficiency, and save resources.

### 🟢 You can view our data for our results here.

### 🟢 Quick Intro
Multimodal Large Language Models (MLLMs) have remarkably progressed in analyzing and understanding images. Despite these advancements, accurately regressing values in charts remains an underexplored area for MLLMs.

### 🟢 What are our pretrained MLLMs?

We used the latest Multimodal Large Language Models technology, such as GPT-4o, Gemini 1.5 Flash, Gemini Pro Vision, and Llama 3.2B Vision.

### 🟢 What are our fine-tuned MLLMs?

We also produced 15 fine-tuned MLLMs (each experiment has three fine-tuned MLLMs) for this study, built on top of Llama 3.2 Vision.

### 🟢 What is our direction for this study?

We would like to compare the performance between fine-tuned and pretrained models to see which models perform better, and also we desire to understand where aspects of MLLMs succeed or fail in the data visualization fields.

### 🟢 What are special features of our study?

We integrated the most advanced packages into our study: Pytorch, Pytorch Lightning, Hugging Face Transformers, Distributed Data Parallel, Vision Encoder Decoder Models, Parameter-Efficient Fine-tuning, Low-Rank Adaptation of Large Language Models, Bits-per-Bite.
We also designed a comprehensive script, and everything in one place, starting from generating datasets, fine-tuning models, and evaluating their MLLMs performances to interpret and analyze visual data.

### 🟢 At this stage, you might consider how to install our project?

We suggest the first step is you’ll need to have the following installed: Python 3.10 or higher, Pip, and Conda.

After you forked and cloned our project into your repo, we recommend using Conda to set up a new environment.

```bash
Conda env create -f condaenvironment.yml
pip install -r piprequirements.txt
