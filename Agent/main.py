import tools
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio
import datetime

provider = OpenAIProvider(
    base_url='',
    api_key=''
)

model = OpenAIChatModel(
    'qwen3-max',
    provider=provider,
)

system_prompt = f"""你是一个功能强大的AI助手，可以通过各种工具帮助用户完成任务。

当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 可用工具分类

### 文件操作
- list_files: 列出目录内容
- read_file: 读取文件内容（支持限制行数）
- write_file: 创建/覆盖文件
- edit_file: 编辑文件（替换指定文本）
- append_file: 追加内容到文件
- copy_file: 复制文件
- rename_file: 重命名/移动文件
- delete_file: 删除文件
- get_file_info: 获取文件详细信息

### 目录操作
- create_directory: 创建目录
- delete_directory: 删除目录

### 搜索
- search_in_files: 在文件中搜索关键词
- search_web: 网络搜索

### 执行
- run_command: 执行Shell/终端命令
- execute_file: 执行脚本文件（Python/JS/Shell等）

### 网络
- fetch_webpage: 抓取网页内容
- http_request: 发送HTTP请求（API调用）

### 实用工具
- get_current_time: 获取当前时间

## 工作原则

1. **先了解再操作**: 操作文件前先用 list_files 和 read_file 了解情况
2. **精确编辑**: 修改文件优先用 edit_file 而不是 write_file 完全覆盖
3. **善用命令行**: run_command 可以执行任意系统命令，非常强大
4. **创建脚本**: 复杂任务可以创建Python脚本来完成
5. **网络信息**: 需要最新信息时使用 search_web 搜索

所有操作都在用户的本地计算机上执行。请用中文回复用户。
## 注意！！用户不会进行任何回复，不要进行任何提问或者让用户回复某个内容
## Attention!! Users will not respond. Do not ask any questions or request any responses from users.
"""

# 注册所有工具
all_tools = [
    # 文件操作
    tools.list_files,
    tools.read_file,
    tools.write_file,
    tools.edit_file,
    tools.append_file,
    tools.copy_file,
    tools.rename_file,
    tools.delete_file,
    tools.get_file_info,
    # 目录操作
    tools.create_directory,
    tools.delete_directory,
    # 搜索
    tools.search_in_files,
    tools.search_web,
    # 执行
    tools.run_command,
    tools.execute_file,
    # 网络
    tools.fetch_webpage,
    tools.http_request,
]

agent = Agent(model,
              deps_type=int,
              system_prompt=system_prompt,
              tools=all_tools)

async def Streaming():
    history = []
    while True:
        user_input = input("Input: ")
        if '新任务' in user_input:
            history = []
        async with agent.run_stream(user_input, message_history=history) as response:
            async for text in response.stream_text(delta=True):
                print(text, end='', flush=True)
            print() 
        history = response.all_messages()

def main():
    history = []
    while True:
        user_input = input("Input: ")
        if '新任务' in user_input:
            history = []
        resp = agent.run_sync(user_input,
                              message_history=history)
        history = list(resp.all_messages())
        print(resp.output)


if __name__ == "__main__":
    main()
    # asyncio.run(Streaming())
