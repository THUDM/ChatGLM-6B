# glm-bot

基于koishi框架的qq聊天机器人


## 环境依赖

* nodejs14以上版本
* gocqhttp

## 使用方法
* 1.启动接口
```
python fastapi.py
```

如果启动的是flask.py
则需要在index.ts文件中将
```
// 启用glm-bot
ctx.plugin(glm_bot,{
    type: 'fastapi',
    myServerUrl: 'http://wx.blockelite.cn:10269/chatglm',
    publicUrl: 'http://127.0.0.1:10269/chat',
    send_glmmtg_response: true,
    prefix: '',
    defaultText: '',
    output: 'quote'
})
```

修改为
```
// 启用glm-bot
ctx.plugin(glm_bot,{
    type: 'flaskapi',
    myServerUrl: 'http://wx.blockelite.cn:10269/chatglm',
    publicUrl: 'http://127.0.0.1:10269/chat',
    send_glmmtg_response: true,
    prefix: '',
    defaultText: '',
    output: 'quote'
})
```

* 2.启动[go-cqhttp](https://github.com/Mrs4s/go-cqhttp)并开启正向ws服务

* 2-1配置onebot
将index.ts中的
```
endpoint: 'ws://127.0.0.1:32333'
```
修改为go-cqhttp的正向ws服务地址

* 3.安装[koishi](https://koishi.chat)依赖

```
cd glm-bot && npm i
```


* 4.启动机器人
```
node -r esbuild-register .
```

## 感谢
* [koishi](https://koishi.chat)


* [go-cqhttp](https://github.com/Mrs4s/go-cqhttp)


* [glm-bot](https://github.com/wochenlong/glm-bot)

* [t4wefan](https://github.com/t4wefan/ChatGLM-6B-with-flask-api)