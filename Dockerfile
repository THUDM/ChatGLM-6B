FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
COPY . .
RUN pip3  install -r requirements.txt
ENV model_path="/model"

EXPOSE 7860

CMD [ "python","web_demo./py" ]