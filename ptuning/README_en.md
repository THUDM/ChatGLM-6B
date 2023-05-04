# ChatGLM-6B-PT
This repository implements tuning of the ChatGLM-6B model based on [P-Tuning v2](https://github.com/THUDM/P-tuning-v2). P-Tuning v2 reduces the amount of parameters that need to be optimized to 0.1% of the full fine-tuning, and then through model quantization, Gradient Checkpoint and other methods, it only needs a minimum of 7GB of video memory to run.

The following uses the [ADGEN](https://aclanthology.org/D19-1321.pdf) (advertising generation) dataset as an example to introduce how to use the code.

## Software dependencies
Running p-tuning requires version 4.27.1 of `transformers`. In addition to the dependencies of ChatGLM-6B, the following dependencies are required
```
pip install rouge_chinese nltk jieba datasets
```
## Instructions

### Download the dataset
The task of the ADGEN dataset is to generate an advertisement word (summary) based on the input (content).

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

From [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) download the processed ADGEN dataset, and put the decompressed `AdvertiseGen` directory into this directory.

### Training

#### P-Tuning v2

Run the following commands for training:
```shell
bash train.sh
```
`PRE_SEQ_LEN` and `LR` in `train.sh` are soft prompt length and training learning rate respectively, which can be adjusted to achieve the best results. The P-Tuning-v2 method will freeze all model parameters, and the quantization level of the original model can be adjusted by adjusting `quantization_bit`. If this option is not added, it will be loaded with FP16 precision.

Under the default configuration of `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, the model parameters of INT4 are frozen, and a training iteration will perform 16 cumulative forward and backward propagations with a batch size of 1, which is equivalent to the total batch size of 16, and only 6.7G GPU memory is required at this time with `quantization_bit=4`. If you want to improve the training efficiency under the same batch size, you can increase the value of `per_device_train_batch_size` while keeping the product of the two unchanged, but it will also bring more GPU memory consumption, please adjust it according to the actual situation.

If you want to [load the model locally](../README_en.md#load-the-model-locally), you can change `THUDM/chatglm-6b` in `train.sh` to your local model path.

#### Finetune
To finetune the full parameters, you need to install [Deepspeed](https://github.com/microsoft/DeepSpeed), and then run the following command:

```shell
bash ds_train_finetune.sh
```

### Inference

During P-tuning v2 training, the model only saves the parameters of the PrefixEncoder part, so the original ChatGLM-6B model and the weight of the PrefixEncoder need to be loaded at the same time during inference, and the arguments need to be specified in `evaluate.sh`:

```shell
--model_name_or_path THUDM/chatglm-6b
--ptuning_checkpoint $CHECKPOINT_PATH
```

It is still compatible with the old version of Checkpoint saved with full parameters, just set `model_name_or_path` as before:

```shell
--model_name_or_path $CHECKPOINT_PATH
```

The evaluation indicators are Chinese Rouge score and BLEU-4. The generated results are saved in
`./output/adgen-chatglm-6b-pt-8-1e-2/generated_predictions.txt`.

### Example
#### Example 1
* Input: 类型#上衣\*材质#牛仔布\*颜色#白色\*风格#简约\*图案#刺绣\*衣样式#外套\*衣款式#破洞
* Label: 简约而不简单的牛仔外套,白色的衣身十分百搭。衣身多处有做旧破洞设计,打破单调乏味,增加一丝造型看点。衣身后背处有趣味刺绣装饰,丰富层次感,彰显别样时尚。
* Output[before tuning]: 这件上衣的材质是牛仔布,颜色是白色,风格是简约,图案是刺绣,衣样式是外套,衣款式是破洞。
* Output[after tuning]: 这是一款简约的牛仔外套,破洞设计,将牛仔布破洞,带来一种随意与个性。破洞的牛仔外套,展现出时尚气息,带来一种休闲感。同时,刺绣图案,让整件外套更加立体。

#### Example 2

* Input: 类型#裙\*版型#显瘦\*风格#文艺\*风格#简约\*图案#印花\*图案#撞色\*裙下摆#压褶\*裙长#连衣裙\*裙领型#圆领
* Label: 文艺个性的印花连衣裙,藏青色底蕴,低调又大气,撞色太阳花分布整个裙身,绚丽而美好,带来时尚减龄的气质。基础款的舒适圆领,简约不失大方,勾勒精致脸庞。领后是一粒包布扣固定,穿脱十分方便。前片立体的打褶设计,搭配后片压褶的做工,增添层次和空间感,显瘦又有型。
* Output[before tuning]: 类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领 1\. 连衣裙:简约风格,裙长为膝盖以上,裙领型为圆领。2\. 裙下摆:压褶设计,使裙摆呈现出流畅的褶皱效果。3\. 裙领型:裙领型为圆领,使穿上连衣裙后更加有型。4\. 版型:采用显瘦设计,让连衣裙看起来更加苗条。5\. 风格:文艺风格,让连衣裙更加有内涵和品味。6\. 图案:印花设计,在连衣裙上印有独特的图案。7\. 撞色:采用撞色设计,让连衣裙在色彩上更加鲜明、富有层次感。
* Output[after tuning]: 这是一款文艺范的连衣裙,以印花为元素,采用简约的印花,既能够突出文艺气质,又能够展现简约风。在印花的同时又有领子和裙摆的压褶设计,更加凸显文艺气质。简约而不会过于单调,搭配出街,穿着十分舒适。

### evaluation result

|               | Finetune | P-tuning v2 | LoRA |
| ------------- | ----------- | ----- | ------------- |
| BLEU-4        | 8.01    | 8.10 | 7.62 |
| Rouge-1       | 31.23  | 31.12 | 30.60 |
| Rouge-2       | 7.36    | 7.11 | 6.96 |
| Rouge-l       | 25.08  | 24.97 | 24.80 |
| Training Loss | 3.00 | 3.74 | 3.32 |

#### Experiment Settings

```
max_source_length=64
max_target_length=64
max_steps=3000
```

##### P-tuning v2

```
pre_seq_len=128
learning_rate=2e-2
quantization_bit=4
per_device_train_batch_size=16
gradient_accumulation_steps=1
```

##### Finetune

```
learning_rate=1e-4
fp16
num_gpus=4
per_device_train_batch_size=4
gradient_accumulation_steps=1
```

##### LoRA

The implementation uses [simple_thu_chatglm6b](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b)

```
learning_rate=5e-4
per_device_train_batch_size=16
gradient_accumulation_steps=1
```

## Model Deployment
First load the tokenizer:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
```

1. If a new Checkpoint needs to be loaded (only contains the PrefixEncoder parameter):

```python
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
     if k.startswith("transformer.prefix_encoder."):
         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
```
Note that you may need to change `pre_seq_len` to the actual value of your training. If you [load model from local](../README_en.md#load-the-model-locally), you need to change `THUDM/chatglm-6b` to the local model path (not the checkpoint path).

2. If you need to load the old checkpoint (including both ChatGLM-6B and PrefixEncoder parameters), or perform full parameter fine-tuning, then directly load the entire checkpoint:

```python
model = AutoModel.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
```

Then it can be quantified according to the needs, or it can be used directly:

```python
# Comment out the following line if you don't use quantization
model = model. quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "Hello", history=[])
```

**[23/04/19]** You can also directly run [web demo](./web_demo.py) which supports loading P-Tuning v2 checkpoint
```shell
bash web_demo.sh
```
It may be necessary to modify the content of [web_demo.sh](./web_demo.sh) to match your actual checkpoint situation.

## Use your own dataset
Modify `train_file`, `validation_file` and `test_file` in `train.sh` and `evaluate.sh` to your own JSON format dataset paths, and change `prompt_column` and `response_column` to the keys in the JSON file corresponding to input text and output text.
You may also need to increase `max_source_length` and `max_target_length` to match the maximum input and output lengths in your own dataset.

## Dialog Dataset

If you need to use multiple rounds of dialogue data to train the model, you can provide chat history. For example, the following is the training data for a three-round dialogue:

```json lines
{"prompt": "长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "response": "用电脑能读数据流吗？水温多少", "history": []}
{"prompt": "95", "response": "上下水管温差怎么样啊？空气是不是都排干净了呢？", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"]]}
{"prompt": "是的。上下水管都好的", "response": "那就要检查线路了，一般风扇继电器是由电脑控制吸合的，如果电路存在断路，或者电脑坏了的话会出现继电器不吸合的情况！", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"], ["95", "上下水管温差怎么样啊？空气是不是都排干净了呢？"]]}
```

During training, you need to specify `--history_column` as the key of the chat history in the data (`history` in this example), and the chat history will be stitched automatically. Note that content exceeding the input length `max_source_length` will be truncated.

You can refer to the following instructions:

```shell
bash train_chat.sh
```

## Citation

```
@inproceedings{liu2022p,
   title={P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks},
   author={Liu, Xiao and Ji, Kaixuan and Fu, Yicheng and Tam, Weng and Du, Zhengxiao and Yang, Zhilin and Tang, Jie},
   booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
   pages={61--68},
   year={2022}
}
```