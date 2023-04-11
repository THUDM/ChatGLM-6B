FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# Downloads to user config dir

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg
# RUN alias python=python3

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl p7zip-full

# Create working directory
RUN rm -rf /usr/src/app && mkdir -p /usr/src/app

ADD ChatGLM.7z /usr/src/app/ChatGLM.7z
RUN cd /usr/src/app/ && 7z x ChatGLM.7z && rm ChatGLM.7z
# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)
# RUN git clone https://github.com/ultralytics/yolov5 /usr/src/yolov5 

# Install pip packages
#COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r /usr/src/app/ChatGLM-6B/requirements.txt
ENV OMP_NUM_THREADS=1
# Cleanup
ENV DEBIAN_FRONTEND teletype
RUN conda install -y cudatoolkit=11.7 -c nvidia && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir streamlit streamlit_chat
EXPOSE 8888
EXPOSE 8501
WORKDIR /usr/src/app/ChatGLM-6B
CMD streamlit run web_demo2.py