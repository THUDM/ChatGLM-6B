## Windows部署

### win10

#### 通过wsl部署

**特别注意**  

对于已在使用的wsl用户，请注意自己数据安全，特别是做好 [备份](https://learn.microsoft.com/zh-cn/windows/wsl/basic-commands#import-and-export-a-distribution)  
对于已在使用的wsl用户，请注意自己数据安全，特别是做好 [备份](https://learn.microsoft.com/zh-cn/windows/wsl/basic-commands#import-and-export-a-distribution)  
对于已在使用的wsl用户，请注意自己数据安全，特别是做好 [备份](https://learn.microsoft.com/zh-cn/windows/wsl/basic-commands#import-and-export-a-distribution)

**常见问题：** 

torch.cuda.OutOfMemoryError: CUDA out of memory.
> 在Windows的系统环境变量中增加  
> 变量名：`PYTORCH_CUDA_ALLOC_CONF`  
> 变量值：`max_split_size_mb:32`  
> 文档书写时使用3090 24G显存配置，其他规格酌情调整 32 至其他值，如未设置变量默认值128极大概率导致 CUDA OOM

无法正常启动：比如命令卡死、无响应、不继续执行等等非报错异常
> 无敌三步走  
> 1.Ctrl+C 终止命令执行，重新执行命令  
> 2.退出实例`exit`,关闭实例`wsl --shutdown`，启动并进入实例 `wsl`  
> 3.重启电脑，重新进入实例，重新执行命令

查看虚拟化是否启用
> 调取任务管理器 `Ctrl+Shift+Esc` 或 `Win+X -> T` 或 `任务栏-> 右键 -> 任务管理器`  
> 性能 -> CPU -> 右下角虚拟化**已启用**  
> 其他方式自行搜索

1. **前置准备**

显卡驱动自行更新 https://www.nvidia.cn/Download/index.aspx?lang=cn 记得备份现有驱动  
wsl开发环境配置 https://learn.microsoft.com/zh-cn/windows/wsl/setup/environment
> 需自行研究，失败因素过多无法列举，尤其是启用虚拟化失败导致无法正常启动wsl服务

wsl导入任意Linux发行版本 https://learn.microsoft.com/zh-cn/windows/wsl/use-custom-distro 可选

2. **wsl安装指定的Linux发行版 Debian**

将 wsl 默认版本设置为 2

```shell
wsl --set-default-version 2
```

安装指定的Linux发行版，本文档以Debian为例

```shell
wsl --install Debian 
```

安装完成后会提示输入新用户名，**直接Ctrl+C** 取消输入操作

默认会安装到C盘，C盘可用空间大于25G，可以跳过迁移步骤，但是 **必须设置wsl默认启动实例** 步骤4

3.**迁移目录**

导出

```shell
 wsl --export Debian E:\debian.tar
```

Debian 实例名  
E:\debian.tar 文件保存的地址

导入

```shell
wsl --import newDebian D:\tools\wsl_data  E:\debian.tar
```

newDebian 新的实例名  
D:\tools\wsl_data 新的实例保存地址  
E:\debian.tar 导入的文件地址

查看导入结果，列出所有实例

```shell
wsl --list --verbose
```

| **NAME** | **STATE** | **VERSION** |
|----------|-----------|-------------|
| *Debian  | Stopped   | 2           |
| newdebian| Stopped   | 2           |

> *代表当前默认启动实例

4. 设置默认环境

修改wsl默认启动实例，**非迁移用户需将 `newDebian` 替换为 `Debian`**

```shell
wsl --set-default newDebian
```

列出所有实例

```shell
wsl --list --verbose
```

| **NAME**   | **STATE** | **VERSION** |
|------------|-----------|-------------|
| Debian     | Stopped   | 2           |
| *newDebian | Stopped   | 2           |

> *代表当前默认启动实例

设置默认Debian默认登录用户为root

```shell
Debian config --default-user root
```

启动Debian

```shell
wsl
```

> 进入 root@DESKTOP-2BUTCG6:/mnt/c/Users/Administrator#

5. 配置环境

更新apt源

```shell
vi /etc/apt/sources.list
```

替换全部
> deb http://mirrors.aliyun.com/debian/ buster main non-free contrib  
deb http://mirrors.aliyun.com/debian/ buster-updates main non-free contrib  
deb http://mirrors.aliyun.com/debian/ buster-backports main non-free contrib  
deb http://mirrors.aliyun.com/debian-security buster/updates main  
deb-src http://mirrors.aliyun.com/debian/ buster main non-free contrib  
deb-src http://mirrors.aliyun.com/debian/ buster-updates main non-free contrib  
deb-src http://mirrors.aliyun.com/debian/ buster-backports main non-free contrib  
deb-src http://mirrors.aliyun.com/debian-security buster/updates main
>
>deb http://mirrors.163.com/debian/ buster main non-free contrib  
deb http://mirrors.163.com/debian/ buster-updates main non-free contrib  
deb http://mirrors.163.com/debian/ buster-backports main non-free contrib  
deb http://mirrors.163.com/debian-security/ buster/updates main non-free contrib  
deb-src http://mirrors.163.com/debian/ buster main non-free contrib  
deb-src http://mirrors.163.com/debian/ buster-updates main non-free contrib  
deb-src http://mirrors.163.com/debian/ buster-backports main non-free contrib  
deb-src http://mirrors.163.com/debian-security/ buster/updates main non-free contrib

更新apt

```shell
apt update
```

安装apt包管理工具

```shell
apt install aptitude  
# y
```

> y 键盘输入，有多个询问则需要按照顺序输入

安装 python3

```shell
aptitude install python3  
# y y
```

安装 python3-pip

```shell
aptitude install python3-pip  
# y y
```

更新 pip

```shell
 pip3 install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple/
```

创建并进入目录

```shell
mkdir /data && cd /data
```

复制文件到当前目录 `/data`

```shell
cp -r /mnt/d/wealth/code/Python/ChatGLM-6B/ .
```

`/mnt/d/wealth/code/Python/ChatGLM-6B/` 项目在Windows系统中所在目录 `/mnt/d` 为`D：`盘
> D:\wealth\code\Python\ChatGLM-6B

`.` 为Debian中的当前目录 `/data`

进入工作目录

```shell
cd ChatGLM-6B
```

安装项目依赖

```shell
pip3 install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

6. 启动项目

启动web版，其他版本安装依赖后可按需启动

```shell
python3 web_demo.py 
```

有一个好消息和一个坏消息  
好消息是：你已经学会启动ChatGLM-6B了！什么时候做大做强干掉ChatGPT？  
坏消息是：漫长的下载开始了...(已经自行下载资源的按照要求放入即可)

7. 代理  
   wsl部署涉及到**非本机访问**请配置代理  
   局域网代理建议用 [nginx](http://nginx.org/)  
   内网穿透建议用 [frp](https://gofrp.org/)
