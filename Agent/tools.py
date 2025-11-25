from pathlib import Path
import os
import subprocess
import datetime
import json
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

base_dir = Path("./WorkDatabase")


def _safe_path(name: str) -> Path:
    """确保路径在base_dir内，防止路径遍历攻击"""
    path = (base_dir / name).resolve()
    if not str(path).startswith(str(base_dir.resolve())):
        raise ValueError("路径越界：不允许访问base_dir之外的文件")
    return path


def read_file(name: str, max_lines: int = None) -> str:
    """
    读取文件内容。
    Parameters:
        name: 文件名/路径
        max_lines: 可选，最大读取行数（防止大文件溢出上下文）
    """
    print(f"(read_file {name}, max_lines={max_lines})")
    try:
        file_path = _safe_path(name)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... 文件已截断，共读取 {max_lines} 行 ...")
                        break
                    lines.append(line)
                content = "".join(lines)
            else:
                content = f.read()
        return content if content else "文件为空"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"读取错误: {e}"

def list_files(directory: str = "") -> str:
    """
    列出目录中的所有文件和文件夹。
    Parameters:
        directory: 可选，子目录路径，默认为根目录
    """
    print(f"(list_files {directory})")
    try:
        target_dir = _safe_path(directory) if directory else base_dir
        if not target_dir.exists():
            return f"错误: 目录 '{directory}' 不存在"
        
        items = []
        for item in sorted(target_dir.iterdir()):
            rel_path = str(item.relative_to(base_dir))
            if item.is_dir():
                items.append(f"📁 {rel_path}/")
            else:
                size = item.stat().st_size
                items.append(f"📄 {rel_path} ({size} bytes)")
        
        return "\n".join(items) if items else "目录为空"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"列出文件错误: {e}"

def rename_file(name: str, new_name: str) -> str:
    """
    重命名或移动文件。
    Parameters:
        name: 原文件名/路径
        new_name: 新文件名/路径
    """
    print(f"(rename_file {name} -> {new_name})")
    try:
        old_path = _safe_path(name)
        new_path = _safe_path(new_name)
        
        if not old_path.exists():
            return f"错误: 文件 '{name}' 不存在"
        
        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(old_path, new_path)
        return f"文件 '{name}' 已重命名为 '{new_name}'"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"重命名错误: {e}"

def delete_file(name: str) -> str:
    """
    删除文件。
    Parameters:
        name: 要删除的文件名/路径
    """
    print(f"(delete_file {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"错误: 文件 '{name}' 不存在"
        os.remove(file_path)
        return f"文件 '{name}' 已删除"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"删除错误: {e}"

def write_file(name: str, content: str) -> str:
    """
    创建或覆盖写入文件。
    Parameters:
        name: 文件名/路径
        content: 要写入的内容
    """
    print(f"(write_file {name})")
    try:
        file_path = _safe_path(name)
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件 '{name}' 写入成功 ({len(content)} 字符)"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"写入错误: {e}"

def execute_file(name: str, args: str = "") -> str:
    """
    执行文件（支持Python、JavaScript、Shell脚本等）。
    Parameters:
        name: 要执行的文件名/路径
        args: 可选，传递给脚本的命令行参数
    """
    print(f"(execute_file {name} {args})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"错误: 文件 '{name}' 不存在"

        # 根据文件扩展名选择执行器
        ext = file_path.suffix.lower()
        executors = {
            ".py": ["python"],
            ".sh": ["bash"],
            ".bat": ["cmd", "/c"],
            ".ps1": ["powershell", "-File"],
            # ".js": ["node"],
        }
        
        if ext not in executors:
            return f"错误: 不支持的文件类型 '{ext}'。支持: {', '.join(executors.keys())}"
        
        cmd = executors[ext] + [str(file_path)]
        if args:
            cmd.extend(args.split())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            cwd=str(base_dir)
        )
        output = result.stdout + result.stderr
        return_code = result.returncode
        return f"返回码: {return_code}\n输出:\n{output}" if output else f"执行完成，返回码: {return_code}"
    except subprocess.TimeoutExpired:
        return "错误: 执行超时（60秒）"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"执行错误: {e}"

def search_web(query: str, max_results: int = 5) -> str:
    """Search web pages. Returns a list of search results (title, link, summary).
    Parameters:
        query: Search keywords
        max_results: Maximum number of results to return, defaults to 5
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
    """
    Fetches webpage content. Can return plain text or HTML content.
    Parameters:
        url: The URL of the webpage to fetch
        extract_text: If True, returns the extracted plain text; if False, returns the raw HTML
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

            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()
            
            text = soup.get_text()
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


def run_command(command: str, timeout: int = 60) -> str:
    """
    执行Shell/终端命令。
    Parameters:
        command: 要执行的命令
        timeout: 超时时间（秒），默认60秒
    """
    print(f"(run_command: {command})")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(base_dir)
        )
        output = result.stdout + result.stderr
        return_code = result.returncode
        return f"返回码: {return_code}\n输出:\n{output}" if output else f"执行完成，返回码: {return_code}"
    except subprocess.TimeoutExpired:
        return f"错误: 命令执行超时（{timeout}秒）"
    except Exception as e:
        return f"执行错误: {e}"


def edit_file(name: str, old_text: str, new_text: str) -> str:
    """
    编辑文件，将old_text替换为new_text（只替换第一次出现）。
    Parameters:
        name: 文件名/路径
        old_text: 要替换的原文本
        new_text: 替换后的新文本
    """
    print(f"(edit_file {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"错误: 文件 '{name}' 不存在"
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if old_text not in content:
            return f"错误: 未找到要替换的文本"
        
        new_content = content.replace(old_text, new_text, 1)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        return f"文件 '{name}' 编辑成功"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"编辑错误: {e}"


def append_file(name: str, content: str) -> str:
    """
    追加内容到文件末尾。
    Parameters:
        name: 文件名/路径
        content: 要追加的内容
    """
    print(f"(append_file {name})")
    try:
        file_path = _safe_path(name)
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"内容已追加到 '{name}'"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"追加错误: {e}"


def search_in_files(keyword: str, file_extension: str = None) -> str:
    """
    在文件中搜索关键词。
    Parameters:
        keyword: 要搜索的关键词
        file_extension: 可选，限制搜索的文件类型，如 ".py", ".txt"
    """
    print(f"(search_in_files keyword='{keyword}', ext={file_extension})")
    results = []
    try:
        for file_path in base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_extension and file_path.suffix != file_extension:
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if keyword.lower() in line.lower():
                            rel_path = file_path.relative_to(base_dir)
                            results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")
            except:
                continue
        
        if results:
            output = f"找到 {len(results)} 处匹配:\n" + "\n".join(results[:50])
            if len(results) > 50:
                output += f"\n... 还有 {len(results) - 50} 处匹配未显示"
            return output
        return "未找到匹配内容"
    except Exception as e:
        return f"搜索错误: {e}"


def create_directory(name: str) -> str:
    """
    创建目录。
    Parameters:
        name: 目录名/路径
    """
    print(f"(create_directory {name})")
    try:
        dir_path = _safe_path(name)
        os.makedirs(dir_path, exist_ok=True)
        return f"目录 '{name}' 创建成功"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"创建目录错误: {e}"


def delete_directory(name: str, force: bool = False) -> str:
    """
    删除目录。
    Parameters:
        name: 目录名/路径
        force: 是否强制删除非空目录
    """
    print(f"(delete_directory {name}, force={force})")
    try:
        import shutil
        dir_path = _safe_path(name)
        if not dir_path.exists():
            return f"错误: 目录 '{name}' 不存在"
        if not dir_path.is_dir():
            return f"错误: '{name}' 不是目录"
        
        if force:
            shutil.rmtree(dir_path)
        else:
            os.rmdir(dir_path)  # 只能删除空目录
        return f"目录 '{name}' 已删除"
    except OSError as e:
        if "not empty" in str(e).lower() or "目录不是空的" in str(e):
            return f"错误: 目录非空，请设置 force=True 强制删除"
        return f"删除错误: {e}"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"删除错误: {e}"


def http_request(url: str, method: str = "GET", data: str = None, headers: str = None) -> str:
    """
    发送HTTP请求（通用API调用）。
    Parameters:
        url: 请求URL
        method: 请求方法 (GET, POST, PUT, DELETE, PATCH)
        data: 请求体数据（JSON字符串格式）
        headers: 请求头（JSON字符串格式）
    """
    print(f"(http_request {method} {url})")
    try:
        req_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json'
        }
        if headers:
            req_headers.update(json.loads(headers))
        
        json_data = json.loads(data) if data else None
        
        response = requests.request(
            method.upper(), 
            url, 
            json=json_data, 
            headers=req_headers, 
            timeout=30
        )
        
        try:
            resp_json = response.json()
            resp_text = json.dumps(resp_json, ensure_ascii=False, indent=2)
        except:
            resp_text = response.text
        
        return f"状态码: {response.status_code}\n响应:\n{resp_text[:8000]}{'...' if len(resp_text) > 8000 else ''}"
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {e}"
    except requests.exceptions.RequestException as e:
        return f"请求错误: {e}"
    except Exception as e:
        return f"错误: {e}"


def get_file_info(name: str) -> str:
    """
    获取文件详细信息（大小、修改时间、行数等）。
    Parameters:
        name: 文件名/路径
    """
    print(f"(get_file_info {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"错误: 文件 '{name}' 不存在"
        
        stat = file_path.stat()
        info = [
            f"文件: {name}",
            f"大小: {stat.st_size} bytes",
            f"修改时间: {datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
            f"创建时间: {datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        # 如果是文本文件，统计行数
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
                info.append(f"行数: {line_count}")
            except:
                pass
        
        return "\n".join(info)
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"获取信息错误: {e}"


def copy_file(source: str, destination: str) -> str:
    """
    复制文件。
    Parameters:
        source: 源文件路径
        destination: 目标文件路径
    """
    print(f"(copy_file {source} -> {destination})")
    try:
        import shutil
        src_path = _safe_path(source)
        dst_path = _safe_path(destination)
        
        if not src_path.exists():
            return f"错误: 源文件 '{source}' 不存在"
        
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return f"文件已复制: '{source}' -> '{destination}'"
    except ValueError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"复制错误: {e}"
