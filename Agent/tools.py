from pathlib import Path
import os
import subprocess
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

base_dir = Path("./test")


def read_file(name: str) -> str:
    """Return file content. If not exist, return error message."""
    print(f"(read_file {name})")
    try:
        with open(base_dir / name, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"An error occurred: {e}"

def list_files() -> list[str]:
    """列出所有文件。返回文件路径列表。"""
    print("(list_file)")
    file_list = []
    for item in base_dir.rglob("*"):
        if item.is_file():
            file_list.append(str(item.relative_to(base_dir)))
    return file_list

def rename_file(name: str, new_name: str) -> str:
    """重命名文件。成功返回成功消息，失败返回错误消息。"""
    print(f"(rename_file {name} -> {new_name})")
    try:
        new_path = base_dir / new_name
        if not str(new_path).startswith(str(base_dir)):
            return "Error: new_name is outside base_dir."

        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(base_dir / name, new_path)
        return f"File '{name}' successfully renamed to '{new_name}'."
    except Exception as e:
        return f"An error occurred: {e}"

def delete_file(name: str) -> str:
    """删除文件。成功返回成功消息，失败返回错误消息。"""
    print(f"(delete_file {name})")
    try:
        file_path = base_dir / name
        if not file_path.exists():
            return f"Error: File '{name}' does not exist."
        os.remove(file_path)
        return f"File '{name}' successfully deleted."
    except Exception as e:
        return f"An error occurred: {e}"

def write_file(name: str, content: str) -> str:
    """创建或写入文件。如果文件存在则覆盖，不存在则创建。成功返回成功消息，失败返回错误消息。"""
    print(f"(write_file {name})")
    try:
        file_path = base_dir / name
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File '{name}' successfully written."
    except Exception as e:
        return f"An error occurred: {e}"

def execute_file(name: str) -> str:
    """执行文件（如Python文件）。返回执行输出或错误消息。"""
    print(f"(execute_file {name})")
    try:
        file_path = base_dir / name
        if not file_path.exists():
            return f"Error: File '{name}' does not exist."
        
        # 根据文件扩展名决定执行方式
        if name.endswith(".py"):
            result = subprocess.run(
                ["python", str(file_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30
            )
            output = result.stdout + result.stderr
            return f"Execution output:\n{output}" if output else "Execution completed with no output."
        else:
            return f"Error: Unsupported file type. Only .py files are supported."
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 30 seconds."
    except Exception as e:
        return f"An error occurred: {e}"

def search_web(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索网页。返回搜索结果列表（标题、链接、摘要）。
    
    参数:
        query: 搜索关键词
        max_results: 返回的最大结果数量，默认为5
    """
    print(f"(search_web query='{query}', max_results={max_results})")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region='cn-zh'))
        
        if not results:
            return "未找到相关搜索结果。"
        
        output = []
        for i, result in enumerate(results, 1):
            title = result.get('title', '无标题')
            link = result.get('href', '无链接')
            snippet = result.get('body', '无摘要')
            output.append(f"{i}. {title}\n   链接: {link}\n   摘要: {snippet}\n")
        
        return "\n".join(output)
    except Exception as e:
        return f"搜索时发生错误: {e}"

def fetch_webpage(url: str, extract_text: bool = True) -> str:
    """抓取网页内容。可以返回纯文本或HTML内容。
    
    参数:
        url: 要抓取的网页URL
        extract_text: 如果为True，返回提取的纯文本；如果为False，返回原始HTML
    """
    print(f"(fetch_webpage url='{url}', extract_text={extract_text})")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        if extract_text:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本和样式元素
            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()
            
            # 获取文本
            text = soup.get_text()
            
            # 清理空白字符
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return f"网页标题: {soup.title.string if soup.title else '无标题'}\n\n内容:\n{text[:5000]}{'...' if len(text) > 5000 else ''}"
        else:
            return response.text[:10000] + ('...' if len(response.text) > 10000 else '')
    
    except requests.exceptions.RequestException as e:
        return f"抓取网页时发生错误: {e}"
    except Exception as e:
        return f"处理网页内容时发生错误: {e}"