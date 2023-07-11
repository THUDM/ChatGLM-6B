from fastapi import FastAPI
import asyncio
import uvicorn
import logging
import logging
import os
import sys
import json
import time


import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
import jieba 
import datasets
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer

from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)



app = FastAPI()

MAX_BATCH_SIZE = 100  # 最大批大小
MAX_WAIT_TIME = 1  # 最大等待时间（秒）

class DataProcessor:
    def __init__(self):
        self.queue = []  # 请求队列
        self.processing = False  # 是否正在进行批量处理
        self.dicts = {}
        self.processing_timer = None  # 定时器对象
        self.event = asyncio.Event()  # 用于通知处理完成的事件

    def process_batch(self):
        while self.queue:
            self.processing = True
            batch = self.queue[:MAX_BATCH_SIZE]
            del self.queue[:MAX_BATCH_SIZE]

            new_batch = predict(batch)
            
            self.dicts.update(dict(zip(batch, new_batch)))
        self.processing = False
        self.event.set()  # 发送处理完成的信号

    async def wait_for_result(self, data):
        while data not in self.dicts:
            await self.event.wait()
            self.event.clear()

    async def process_data(self, data):
        self.queue.append(data)

        if len(self.queue) == 1:
            await asyncio.sleep(MAX_WAIT_TIME)
            if not self.processing:
                self.process_batch()

        elif len(self.queue) >= MAX_BATCH_SIZE and not self.processing:
            self.process_batch()
        
        await self.wait_for_result(data)
        # logging.info(self.dicts)
        return json.loads(self.dicts[data])

data_processor = DataProcessor()

@app.get("/data")
async def handle_data(prompt: str):
    return await data_processor.process_data(prompt)



def get_trainer_tokenizer(model_args, data_args, training_args):

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=True
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )
    return trainer,tokenizer

def predict(prompts):
    
    print('*'*50)
    global trainer, tokenizer
    data = {
        "instruction": prompts,
        "output": [1]*len(prompts)
        }
    
    predict_dataset = datasets.Dataset.from_dict(data)

    def preprocess_function_eval(examples):
        prompt_column = 'instruction'
        inputs = []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] :
                inputs.append(examples[prompt_column][i])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        return model_inputs
    
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
         predict_dataset = predict_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
         )

    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
    predictions = tokenizer.batch_decode(
    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    return predictions


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    trainer, tokenizer = get_trainer_tokenizer(model_args, data_args, training_args)

    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8002)
