FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04
LABEL maintainer="steinven@outlook.it"
LABEL version="1.0"
LABEL description="ChatGLM-6B"

#github网速太慢或被墙,使用码云加速
ENV REPO_LINK=https://gitee.com/xiaoyaolangzi/ChatGLM-6B.git


#安装所需的库文件
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list;apt update;apt install git -y \
	&& apt install  python3 python3-pip -y \
	&& pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple \
	&& git clone --depth=1 $REPO_LINK /ChatGLM-6B \
    && pip install -r /ChatGLM-6B/requirements.txt

WORKDIR /ChatGLM-6B
EXPOSE 7860

CMD ["python3","web_demo.py"]

#可以根据将web_demo.py下载到宿主机，通过挂载的方式，修改参数
#运行示例：docker run -it --gpus all -p 7860:7860  -v $PWD/web_demo.py:/ChatGLM-6B/web_demo.py IMAGE_ID