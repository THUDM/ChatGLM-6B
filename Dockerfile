FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
COPY . .
RUN pip3  install -r requirements.txt
RUN python3 pull_model.py
EXPOSE 7860
CMD [ "python","web_demo.py" ]