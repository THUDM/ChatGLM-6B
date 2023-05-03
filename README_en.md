# ChatGLM-6B

<p align="center">
   🌐 <a href="https://chatglm.cn/blog" target="_blank">Blog</a> • 🤗 <a href="https://huggingface.co/THUDM/chatglm-6b" target="_blank">HF Repo</a> • 🐦 <a href="https://twitter.com/thukeg" target="_blank">Twitter</a> • 📃 <a href="https://arxiv.org/abs/2103.10360" target="_blank">[GLM@ACL 22]</a> <a href="https://github.com/THUDM/GLM" target="_blank">[GitHub]</a> • 📃 <a href="https://arxiv.org/abs/2210.02414" target="_blank">[GLM-130B@ICLR 23]</a> <a href="https://github.com/THUDM/GLM-130B" target="_blank">[GitHub]</a> <br>
</p>
<p align="center">
    👋 Join our <a href="https://join.slack.com/t/chatglm/shared_invite/zt-1th2q5u69-7tURzFuOPanmuHy9hsZnKA" target="_blank">Slack</a> and <a href="resources/WECHAT.md" target="_blank">WeChat</a>
</p>

## Introduction

ChatGLM-6B is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level).

ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue. The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning, feedback bootstrap, and reinforcement learning wit human feedback. With only about 6.2 billion parameters, the model is able to generate answers that are in line with human preference.

In order to facilitate downstream developers to customize the model for their own application scenarios, we also implements an parameter-efficient tuning method based on [P-Tuning v2](https://github.com/THUDM/P-tuning-v2)[(Guidelines)](ptuning/README_en.md). Tuning requires at least 7GB of GPU memory at INT4 quantization level.

Try the [online demo](https://huggingface.co/spaces/ysharma/ChatGLM-6b_Gradio_Streaming) on Huggingface Spaces.

## Projects
Open source projects that accelerate ChatGLM:
* [ChatGLM-MNN](https://github.com/wangzhaode/ChatGLM-MNN): An MNN-based implementation of ChatGLM-6B C++ inference, which supports automatic allocation of computing tasks to GPU and CPU according to the size of GPU memory
* [JittorLLMs](https://github.com/Jittor/JittorLLMs): Running ChatGLM-6B in FP16 with a minimum of 3G GPU memory or no GPU at all, with Linux, windows, and Mac support

Open source projects using ChatGLM-6B:
* [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM): ChatGLM application based on langchain, realizing Q&A based on extensible knowledge base
* [Wenda](https://github.com/l15y/wenda): Large-scale language model call platform, based on ChatGLM-6B to achieve ChatPDF-like functions
* [chatgpt_academic](https://github.com/binary-husky/chatgpt_academic): An academic writing and programming toolbox that supports ChatGLM-6B. It has the characteristics of modularization and multi-thread calling LLM, and can call multiple LLMs in parallel.
* [glm-bot](https://github.com/initialencounter/glm-bot): Connect ChatGLM to Koishi to call ChatGLM on major chat platforms

Example projects supporting online training of ChatGLM-6B and related applications:
* [ChatGLM-6B deployment and fine-tuning tutorial](https://www.heywhale.com/mw/project/6436d82948f7da1fee2be59e)
* [ChatGLM-6B combined with langchain to implement local knowledge base QA Bot](https://www.heywhale.com/mw/project/643977aa446c45f4592a1e59)

Third-party evaluation:
* [Measuring Massive Multitask Chinese Understanding](https://arxiv.org/abs/2304.12986)

For more open source projects, see [PROJECT.md](PROJECT.md)

## Getting Started

### Hardware Requirements

| **Quantization Level** | **GPU Memory** |
|------------------------|----------------|
| FP16（no quantization）  | 13 GB          |
| INT8                   | 10 GB          |
| INT4                   | 6 GB           |

### Environment Setup

Install the requirements with pip: `pip install -r requirements.txt`. `transformers` library version is recommended to be `4.27.1`, but theoretically any version no lower than `4.23.1` is acceptable.

In addition, if you need to run the quantified model on the CPU, you also need to install `gcc` and `openmp`. Most Linux distributions are installed by default. For Windows, you can check `openmp` when installing [TDM-GCC](https://jmeubank.github.io/tdm-gcc/). On Windows testing environment, the `gcc` version is `TDM-GCC 10.3.0`, and on Linux is `gcc 11.3.0`.

### Usage

Generate dialogue with the following code

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:

1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。

如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```
The implementation of the model is still in development. If you want to fix the used model implementation to ensure compatibility, you can add the `revision="v0.1.0"` parameter in the `from_pretrained` call. `v0.1.0` is the latest version number. For a complete list of versions, see [Change Log](https://huggingface.co/THUDM/chatglm-6b#change-log).

### Load the model locally
The above code will automatically download the model implementation and checkpoints by [transformers](https://github.com/huggingface/transformers). The full model implementation can be found at [Hugging Face Hub](https://huggingface.co/THUDM/chatglm-6b). If your network environment is poor, downloading model parameters may take a long time or even fail. At this point, you can download the model to the local first, and then load it from the local.

To download models from Hugging Face Hub, you need to [install Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) , then run
```Shell
git clone https://huggingface.co/THUDM/chatglm-6b
```

After downloading the model locally, replace `THUDM/chatglm-6b` in the above code with the path of your local `chatglm-6b` folder to load the model locally.

**Optional**: The implementation of the model is still in development. If you want to fix the used model implementation to ensure compatibility, you can execute
```Shell
git checkout v0.1.0
```

## Demo & API

We provide a Web demo based on [Gradio](https://gradio.app) and a command line demo in the repo. First clone our repo with:

```shell
git clone https://github.com/THUDM/ChatGLM-6B
cd ChatGLM-6B
```

### Web Demo

![web-demo](resources/web-demo.gif)

Install Gradio `pip install gradio`，and run [web_demo.py](web_demo.py):

```shell
python web_demo.py
```

The program runs a web server and outputs the URL. Open the URL in the browser to use the web demo.

Thanks to [@AdamBear](https://github.com/AdamBear) for implementing a web demo based on Streamlit, see [#117](https://github.com/THUDM/ChatGLM-6B/pull/117 ).

#### CLI Demo

![cli-demo](resources/cli-demo.png)

Run [cli_demo.py](cli_demo.py) in the repo:

```shell
python cli_demo.py
```

The command runs an interactive program in the shell. Type your instruction in the shell and hit enter to generate the response. Type `clear` to clear the dialogue history and `stop` to terminate the program.

## API Deployment
First install the additional dependency `pip install fastapi uvicorn`. The run [api.py](api.py) in the repo.
```shell
python api.py
```
By default the api runs at the`8000`port of the local machine. You can call the API via 
```shell
curl -X POST "http://127.0.0.1:8000" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好", "history": []}'
```
The returned value is
```shell
{
  "response":"你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
  "history":[["你好","你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"]],
  "status":200,
  "time":"2023-03-23 21:38:40"
}
```

## Deployment

### Quantization

By default, the model parameters are loaded with FP16 precision, which require about 13GB of GPU memory. It your GPU memory is limited, you can try to load the model parameters with quantization:

```python
# Change according to your hardware. Only support 4/8 bit quantization now.
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(8).cuda()
```

After 2 to 3 rounds of dialogue, the GPU memory usage is about 10GB under 8-bit quantization, and only 6GB under 4-bit quantization. As the number of dialogue rounds increases, the corresponding GPU memory consumption also increases. Due to the use of relative position encoding, ChatGLM-6B theoretically supports an infinitely long context-length, but the performance will gradually decline after the total length exceeds 2048 (training length).

Model quantization brings a certain performance decline. After testing, ChatGLM-6B can still perform natural and smooth generation under 4-bit quantization. using [GPT-Q](https://arxiv.org/abs/2210.17323) etc. The quantization scheme can further compress the quantization accuracy/improve the model performance under the same quantization accuracy. You are welcome to submit corresponding Pull Requests.

The quantization costs about 13GB of CPU memory to load the FP16 model. If your CPU memory is limited, you can directly load the quantized model, which costs only 5.2GB CPU memory: 
```python
# For INT8-quantized model, change "chatglm-6b-int4" to "chatglm-6b-int8"
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
```

### CPU Deployment

If your computer is not equipped with GPU, you can also conduct inference on CPU, but the inference speed is slow (and taking about 32GB of memory):

```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
```

If your CPU memory is limited, you can directly load the quantized model:
```python
# For INT8-quantized model, change "chatglm-6b-int4" to "chatglm-6b-int8"
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).float()
```

If your encounter the error `Could not find module 'nvcuda.dll'` or `RuntimeError: Unknown platform: darwin`(MacOS), please [load the model locally](README_en.md#load-the-model-locally). 


### CPU Deployment on Mac

The default Mac enviroment does not support Openmp. One may encounter such warning/errors when execute the `AutoModel.from_pretrained(...)` command:

```sh
clang: error: unsupported option '-fopenmp'
clang: error: unsupported option '-fopenmp'
```

Take the quantified int4 version [chatglm-6b-int4](https://huggingface.co/THUDM/chatglm-6b-int4) for example, the following extra steps are needed:

1. Install `libomp`;
2. Configure `gcc`.

```bash
# STEP 1: install libopenmp, check `https://mac.r-project.org/openmp/` for details
## Assumption: `gcc -v >= 14.x`, read the R-Poject before run the code:
curl -O https://mac.r-project.org/openmp/openmp-14.0.6-darwin20-Release.tar.gz
sudo tar fvxz openmp-14.0.6-darwin20-Release.tar.gz -C /
## Four files are installed:
#   usr/local/lib/libomp.dylib
#   usr/local/include/ompt.h
#   usr/local/include/omp.h
#   usr/local/include/omp-tools.h
```

For `chatglm-6b-int4`, modify the [quantization.py](https://huggingface.co/THUDM/chatglm-6b-int4/blob/main/quantization.py)file. In the file, change the `gcc -O3 -fPIC -pthread -fopenmp -std=c99` configuration to `gcc -O3 -fPIC -Xclang -fopenmp -pthread  -lomp -std=c99`，[corresponding python code](https://huggingface.co/THUDM/chatglm-6b-int4/blob/63d66b0572d11cedd5574b38da720299599539b3/quantization.py#L168), i.e.:

```python
# STEP
## Change the line contains `gcc -O3 -fPIC -pthread -fopenmp -std=c99` to:
compile_command = "gcc -O3 -fPIC -Xclang -fopenmp -pthread  -lomp -std=c99 {} -shared -o {}".format(source_code, kernel_file)
```

For production code, one could use `platform` library to make it neater:

```python
## import platform just after `import os`
import platform
## ...
## change the corresponding lines to:
if platform.uname()[0] == 'Darwin':
    compile_command = "gcc -O3 -fPIC -Xclang -fopenmp -pthread  -lomp -std=c99-o {}".format(
    source_code, kernel_file)
else:
    compile_command = "gcc -O3 -fPIC -pthread -fopenmp -std=c99 {} -shared -o {}".format(
    source_code, kernel_file)
```

> Notice: If you have run the `chatglm` project and failed, you may want to clean the cache of Huggingface before your next try, i.e. `rm -rf ${HOME}/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4`. Since `rm` is used, please MAKE SURE that the command deletes the right files.

### GPU Inference on Mac
For Macs (and MacBooks) with Apple Silicon, it is possible to use the MPS backend to run ChatGLM-6B on the GPU. First, you need to refer to Apple's [official instructions](https://developer.apple.com/metal/pytorch) to install PyTorch-Nightly.

Currently you must [load the model locally](README_en.md#load-the-model-locally) on MacOS. Change the code to load the model from your local path, and use the mps backend:
```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).half().to('mps')
```

For Mac users with Mac >= 13.3, one may encounter errors related to `half()` method. Use `float()` instead:

```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).float().to('mps')
```

Then you can use GPU-accelerated model inference on Mac.

> Notice: there is no promblem to run the non-quantified version of ChatGLM with MPS. One could check [this issue](https://github.com/THUDM/ChatGLM-6B/issues/462) to run the quantified version with MPS as the backend (and get `ERRORS`). Unzip/unpack [kernel](https://huggingface.co/THUDM/chatglm-6b/blob/658202d88ac4bb782b99e99ac3adff58b4d0b813/quantization.py#L27) as an `ELF` file shows its backend is `cuda`.

### Multi-GPU Deployment
If you have multiple GPUs, but the memory size of each GPU is not sufficient to accommodate the entire model, you can split the model across multiple GPUs. 

First, install accelerate: `pip install accelerate`, and then load the model using the following method:
```python
from utils import load_model_on_gpus
model = load_model_on_gpus("THUDM/chatglm-6b", num_gpus=2)
```

This will deploy the model onto two GPUs for inference. You can change `num_gpus` to the number of GPUs you want to use. By default, the model is split evenly, but you can also specify the `device_map` parameter to customize the splitting.

## Parameter-efficient Tuning
Parameter-efficient tuning based on [P-tuning v2](https://github.com/THUDM/P-tuning-v2). See [ptuning/README.md](ptuning/README.md) for details on how to use it.

## Update
**[2023/04/16]** Added INT8 quantized model [ChatGLM-6B-INT8](https://huggingface.co/THUDM/chatglm-6b-int8). Added multi-GPU deployment (thanks to [@Cherrysaber](https://github.com/Cherrysaber)).

**[2023/04/06]** Improve the web demo interface (thanks to [@tuteng0915](https://github.com/tuteng0915)). Remove the image tokens in the embedding layer to reduce the memory usage (need to update the model files `pytorch_model-00001-of-00008.bin` and `pytorch_model-00008-of-00008.bin`, thanks to [@silverriver](https:/ /github.com/silverriver) for proposing the idea). Removed dependency on `icetk` (need to update model file `ice_text.model`).

**[2023/03/31]** Added a parameter-efficient tuning implementation based on [P-Tuning-v2](https://github.com/THUDM/P-tuning-v2). The minimum INT4 quantization level only needs 7GB GPU memory is enough for model tuning. See [Parameter-efficient tuning method](ptuning/README.md) for details.

**[2023/03/23]** Add API deployment, thanks to [@LemonQu-GIT](https://github.com/LemonQu-GIT). Add embedding-quantized model [ChatGLM-6B-INT4-QE](https://huggingface.co/THUDM/chatglm-6b-int4-qe). Add support for GPU inference on Mac with Apple Silicon.

**[2023/03/19]** Add streaming output function `stream_chat`, already applied in web and CLI demo. Fix Chinese punctuations in output. Add quantized model [ChatGLM-6B-INT4](https://huggingface.co/THUDM/chatglm-6b-int4). 

## ChatGLM-6B Examples

The following are some Chinese examples with `web_demo.py`. Welcome to explore more possibility with ChatGLM-6B.

<details><summary><b>Self Cognition</b></summary>

![](examples/self-introduction.png)

</details>

<details><summary><b>Outline</b></summary>

![](examples/blog-outline.png)

</details>

<details><summary><b>Ad</b></summary>

![](examples/ad-writing-2.png)

![](examples/comments-writing.png)

</details>

<details><summary><b>Email</b></summary>

![](examples/email-writing-1.png)

![](examples/email-writing-2.png)

</details>

<details><summary><b>Information Extraction</b></summary>

![](examples/information-extraction.png)

</details>

<details><summary><b>Role Play</b></summary>

![](examples/role-play.png)

</details>

<details><summary><b>Comparison</b></summary>

![](examples/sport.png)

</details>

<details><summary><b>Travel Guide</b></summary>

![](examples/tour-guide.png)

</details>

## License

This repository is licensed under the [Apache-2.0 License](LICENSE). The use of ChatGLM-6B model weights is subject to the [Model License](MODEL_LICENSE)。

## Citation

If you find our work useful, please consider citing the following papers:

```
@inproceedings{
  zeng2023glm-130b,
  title={{GLM}-130B: An Open Bilingual Pre-trained Model},
  author={Aohan Zeng and Xiao Liu and Zhengxiao Du and Zihan Wang and Hanyu Lai and Ming Ding and Zhuoyi Yang and Yifan Xu and Wendi Zheng and Xiao Xia and Weng Lam Tam and Zixuan Ma and Yufei Xue and Jidong Zhai and Wenguang Chen and Zhiyuan Liu and Peng Zhang and Yuxiao Dong and Jie Tang},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=-Aw0rrrPUF}
}
```

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
