import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel

import inquirer
import torch
# 参数
choices_jobType = [("GPU", 1), ("CPU", 2)]
choices_floatType = [("half", 1), ("float", 2)]
choices_model = [("默认(chatglm-6b)", 'chatglm-6b'), ("量化int4(chatglm-6b-int4)", 'chatglm-6b-int4')]

def print_list(choices, v):
    for element in choices:
        if element[1] == v:
            return element[0]
    return None


def print_confirm(v):
    if v:
        return '是'
    else:
        return '否'


def print_confirm2(display, v1, v2=True, v3=True):
    if v1 and v2 and v3:
        return display
    else:
        return ''

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main(answers):
    model_name = answers['path'] + answers['model']

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    # 精度设置
    if answers['float_type'] == 2:
        model = model.float()
    else:
        model = model.half()
    # 设备设置
    if answers['job_type'] == 1:
        if os_name == 'Darwin':
            model = model.to("mps")
        else:
            model = model.cuda()

    model = model.eval()





    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    isGPUSport = torch.cuda.is_available() or (torch.backends.mps.is_available() if os_name == 'Darwin' else False)
    # 设置选项
    questions = [
        inquirer.List(
            "job_type",
            message="选择运行类型?",
            default=1 if isGPUSport else 2,
            choices=choices_jobType,
            # 如果支持GPU,默认GPU
            # 如果不支持GPU的话，默认CPU，不显示
            ignore= not isGPUSport,
        ),
        inquirer.List(
            "float_type",
            message="选择浮点精度?",
            # mac mps半精度容易报错，默认float
            # 默认使用half
            default=2 if os_name == 'Darwin' else 1,
            choices=choices_floatType,

        ),
        inquirer.Confirm(
            "isLocal",
            message="是否使用本地模型",
            default=True,
        ),
        inquirer.Text(
            "path",
            message="设置模型路径",
            # 使用本地模型的话，可以设置目录
            default=lambda answer: './models/' if answer['isLocal'] else 'THUDM/',
            ignore=lambda answer: not answer['isLocal'],
        ),
        inquirer.List(
            "model",
            message="选择模型?",
            # mac mps半精度容易报错，默认float
            # 默认使用half
            default='chatglm-6b' if os_name == 'Darwin' else 'chatglm-6b-int4',
            choices=choices_model,
            ignore=os_name == 'Darwin',
        ),

    ]

    # 处理选项
    answers = inquirer.prompt(questions)

    print('========= 选项 =========')
    print('运行类型: %s' % (print_list(choices_jobType, answers['job_type'])))
    print('浮点精度: %s' % (print_list(choices_floatType, answers['float_type'])))
    print('本地模型: %s' % (print_confirm(answers['isLocal'])))
    print('模型: %s%s' % (answers['path'],  answers['model']))
    if os_name == 'Darwin':
        print('----说明-----')
        print('MacOS下，如果使用GPU报错的话，建议：')
        print('1.安装 PyTorch-Nightly：pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu')
        print('2.出现 LLVM ERROR: Failed to infer result type(s). 可以把精度设置为float')
    print('------------------------')
    main(answers)
