## Q1

**Mac直接加载量化后的模型出现提示 `clang: error: unsupported option '-fopenmp'**

这是由于Mac由于本身缺乏omp导致的，此时可运行但是单核。需要单独安装 openmp 依赖，即可在Mac下使用OMP：

```bash
# 参考`https://mac.r-project.org/openmp/`
## 假设: gcc(clang)是14.x版本，其他版本见R-Project提供的表格
curl -O https://mac.r-project.org/openmp/openmp-14.0.6-darwin20-Release.tar.gz
sudo tar fvxz openmp-14.0.6-darwin20-Release.tar.gz -C /
```
此时会安装下面几个文件：`/usr/local/lib/libomp.dylib`, `/usr/local/include/ompt.h`, `/usr/local/include/omp.h`, `/usr/local/include/omp-tools.h`。

> 注意：如果你之前运行`ChatGLM`项目失败过，最好清一下Huggingface的缓存，i.e. 默认下是 `rm -rf ${HOME}/.cache/huggingface/modules/transformers_modules/chatglm-6b-int4`。由于使用了`rm`命令，请明确知道自己在删除什么。