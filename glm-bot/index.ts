import { Context } from 'koishi'
import console from '@koishijs/plugin-console'
import * as sandbox from '@koishijs/plugin-sandbox'
import * as echo from '@koishijs/plugin-echo'

import onebot from '@koishijs/plugin-adapter-onebot'

import glm_bot from './glm-bot'

// 创建一个 Koishi 应用
const ctx = new Context({
  port: 5140,
})
// 使用 OneBot 适配器的机器人
ctx.plugin(onebot, {
    protocol: 'ws',
    selfId: '3111720341',
    endpoint: 'ws://127.0.0.1:32333',
  })

// 启用上述插件
ctx.plugin(console)     // 提供控制台
ctx.plugin(sandbox)     // 提供调试沙盒
ctx.plugin(echo)        // 提供回声指令

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

// 启动应用
ctx.start()