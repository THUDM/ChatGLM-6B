# nvidia/cuda: https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=devel-ubuntu18.04
# 1. check host nvidia drvier version: nvidia-smi
#           Nvidia driver version compatible cuda version: https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility
#                   CUDA 12.x: Linux x86_64 minimum driver version >= 525.60.13
#                   CUDA 11.x: Linux x86_64 minimum driver version >= 450.80.02
# 2. check target PyTorch version: import torch; print(torch.cuda.is_available()) -> True
#           PyTorch version compatible cuda version: https://pytorch.org/get-started/previous-versions/
FROM nvidia/cuda:11.2.2-devel-ubuntu18.04

# set timezone
RUN apt-get update -y && apt-get install -y tzdata
ENV TZ=Asia/Shanghai

# install miniconda: https://docs.conda.io/en/latest/miniconda.html#linux-installers
RUN apt-get -y update && apt-get install -y wget && \
    mkdir -p /app && wget -P /app https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    cd /app && bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b -p /app/dev/miniconda3 && \
    source /app/dev/miniconda3/bin/activate base && \
    conda init bash && source ~/.bashrc && conda env list \
    rm /app/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && ls -al /app/

# install chatglm-6b dependencies
RUN source /app/dev/miniconda3/bin/activate base && conda env list && \
    pip install protobuf \
                torch transformers datasets accelerate peft \
                icetk cpm_kernels rouge_chinese nltk jieba \
                gradio==3.20.0 mdtex2html fastapi uvicorn requests
RUN mkdir -p /app/source && cd /app/source && \
    apt-get update -y && apt-get install -y git && \
    git clone --recursive https://github.com/zealotpb/ChatGLM-6B.git && \
    cd /app/source/ChatGLM-6B/ptuning && \
    apt-get update -y && apt-get install -y git-lfs && git lfs install && \
    git clone --recursive https://huggingface.co/THUDM/chatglm-6b && ls -al chatglm-6b/

# copy start.sh
COPY start.sh /usr/bin/start
RUN chmod a+x /usr/bin/start

# install necessary debug tools
RUN apt-get update -y && apt-get install -y vim htop

CMD ["/bin/bash"]
