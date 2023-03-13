import os
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

history = []
while True:
    query = input("请输入提示，clear清空对话历史，stop终止程序：\n")
    if query == "stop":
        break
    if query == "clear":
        history = []
        os.system('clear')
        continue
    print("回复：")
    response, history = model.chat(tokenizer, query, history=history)
    print(response)
