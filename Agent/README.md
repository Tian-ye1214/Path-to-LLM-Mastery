一个基于 [Pydantic AI](https://ai.pydantic.dev/) 框架的智能 Agent 示例项目，**核心理念**：让 AI Agent 不仅能使用工具，更能创造工具，实现真正的自主问题解决能力。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pydantic-ai
pip install ddgs
pip install requests
pip install beautifulsoup4
```

### 2. 配置 API Key

在 `main.py` 中修改 DeepSeek API Key：

```python
model = OpenAIChatModel(
    'deepseek-reasoner',
    provider=DeepSeekProvider(api_key='你的API_KEY'),
)
```

### 3. 运行

```bash
python main.py
```

## 📦 项目结构

```
Agent/
├── main.py          # 主程序入口，Agent 配置和对话循环
├── tools.py         # 工具集定义（文件操作、网络搜索等）
├── test/            # Agent 工作目录（创建和执行文件的位置）
└── README.md        # 项目文档
```

## 🛠️ 内置工具

| 工具名称 | 功能描述 |
|---------|---------|
| `read_file` | 读取文件内容 |
| `write_file` | **创建或覆盖文件（核心工具）** |
| `execute_file` | **执行 Python 文件（核心工具）** |
| `list_files` | 列出所有文件 |
| `rename_file` | 重命名文件 |
| `delete_file` | 删除文件 |
| `search_web` | DuckDuckGo 网络搜索 |
| `fetch_webpage` | 抓取网页内容 |


## 🌟 未来思考

- 添加更多编程语言的执行支持（JavaScript、Shell 等）
- 实现工具的持久化和复用机制