from pathlib import Path
import os
import subprocess
import datetime
import json_repair as json
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import MultimodalTools
import logger

base_dir = Path("./WorkDatabase")


def _safe_path(name: str) -> Path:
    """Ensure path is within base_dir to prevent path traversal attacks"""
    path = (base_dir / name).resolve()
    if not str(path).startswith(str(base_dir.resolve())):
        raise ValueError("Path traversal detected: access outside base_dir is not allowed")
    return path


def read_file(name: str, max_lines: int = None) -> str:
    """
    Read file contents.
    Parameters:
        name: File name/path
        max_lines: Optional, maximum number of lines to read (prevents context overflow for large files)
    """
    logger.debug(f"(read_file {name}, max_lines={max_lines})")
    try:
        file_path = _safe_path(name)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... File truncated, read {max_lines} lines ...")
                        break
                    lines.append(line)
                content = "".join(lines)
            else:
                content = f.read()
        return content if content else "File is empty"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Read error: {e}"

def list_files(directory: str = "") -> str:
    """
    List all files and folders in a directory.
    Parameters:
        directory: Optional, subdirectory path, defaults to root directory
    """
    logger.debug(f"(list_files {directory})")
    try:
        target_dir = _safe_path(directory) if directory else base_dir
        if not target_dir.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        items = []
        for item in sorted(target_dir.iterdir()):
            rel_path = str(item.relative_to(base_dir))
            if item.is_dir():
                items.append(f"{rel_path}/")
            else:
                size = item.stat().st_size
                items.append(f"{rel_path} ({size} bytes)")
        
        return "\n".join(items) if items else "Directory is empty"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error listing files: {e}"

def rename_file(name: str, new_name: str) -> str:
    """
    Rename or move a file.
    Parameters:
        name: Original file name/path
        new_name: New file name/path
    """
    logger.debug(f"(rename_file {name} -> {new_name})")
    try:
        old_path = _safe_path(name)
        new_path = _safe_path(new_name)
        
        if not old_path.exists():
            return f"Error: File '{name}' does not exist"
        
        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(old_path, new_path)
        return f"File '{name}' has been renamed to '{new_name}'"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Rename error: {e}"

def delete_file(name: str) -> str:
    """
    Delete a file.
    Parameters:
        name: File name/path to delete
    """
    logger.debug(f"(delete_file {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"File '{name}' does not exist"
        os.remove(file_path)
        return f"File '{name}' has been deleted"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Delete error: {e}"

def write_file(name: str, content: str) -> str:
    """
    Create or overwrite a file.
    Parameters:
        name: File name/path (relative to WorkDatabase directory)
        content: Content to write (must be a string)
    
    IMPORTANT LIMITATIONS:
    - Maximum content length: 10000 characters (to avoid JSON parsing issues)
    - For larger files: Use write_file_chunked() or split content into multiple writes
    - For code files: Keep under 300 lines per file for best results
    """
    content_len = len(content) if content else 0
    logger.debug(f"(write_file {name}, content_length={content_len}) [开始]")
    try:
        if content is None:
            return "Write error: content cannot be None"
        if not isinstance(content, str):
            content = str(content)

        line_count = content.count('\n') + 1
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = _safe_path(name)
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        result = f"File '{name}' written successfully ({len(content)} characters, {line_count} lines)"
        return result
    except ValueError as e:
        return f"Security error: {e}"
    except PermissionError as e:
        return f"Permission error: Cannot write to '{name}' - {e}"
    except Exception as e:
        return f"Write error: {type(e).__name__} - {e}"

def execute_file(name: str, args: str = "") -> str:
    """
    Execute a file (supports Python, Shell scripts, etc.).
    Parameters:
        name: File name/path to execute
        args: Optional, command-line arguments to pass to the script
    """
    logger.debug(f"(execute_file {name} {args})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"Error: File '{name}' does not exist"

        ext = file_path.suffix.lower()
        executors = {
            ".py": ["python"],
            ".sh": ["bash"],
            ".bat": ["cmd", "/c"],
            ".ps1": ["powershell", "-File"],
            # ".js": ["node"],
        }
        
        if ext not in executors:
            return f"Error: Unsupported file type '{ext}'. Supported: {', '.join(executors.keys())}"
        
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
        return f"Return code: {return_code}\nOutput:\n{output}" if output else f"Execution completed, return code: {return_code}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out (60 seconds)"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Execution error: {e}"

def search_web(query: str, max_results: int = 5) -> str:
    """
    Search web pages. Returns a list of search results (title, link, summary).
    Parameters:
        query: Search keywords
        max_results: Maximum number of results to return, defaults to 5
    """
    logger.debug(f"(search_web query='{query}', max_results={max_results})")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region='cn-zh'))
        
        if not results:
            return "No relevant search results found."
        
        output = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            link = result.get('href', 'No link')
            snippet = result.get('body', 'No summary')
            output.append(f"{i}. {title}\n   Link: {link}\n   Summary: {snippet}\n")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error during search: {e}"

def fetch_webpage(url: str, extract_text: bool = True) -> str:
    """
    Fetch webpage content. Can return plain text or HTML content.
    Parameters:
        url: The URL of the webpage to fetch
        extract_text: If True, returns the extracted plain text; if False, returns the raw HTML
    """
    logger.debug(f"(fetch_webpage url='{url}', extract_text={extract_text})")
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
            
            return f"Page Title: {soup.title.string if soup.title else 'No title'}\n\nContent:\n{text[:5000]}{'...' if len(text) > 5000 else ''}"
        else:
            return response.text[:10000] + ('...' if len(response.text) > 10000 else '')
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching webpage: {e}"
    except Exception as e:
        return f"Error processing webpage content: {e}"


import shlex
import platform as _platform

_DANGEROUS_PATTERNS = [
    'rm -rf /',
    'rm -rf /*',
    'mkfs.',
    'dd if=',
    ':(){:|:&};:',
    '> /dev/sda',
    'chmod -R 777 /',
    'chown -R',
    '| sh',
    '| bash',
    '`',
    '$(',
    'eval ',
    'exec ',
]

def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check if command contains dangerous patterns"""
    command_lower = command.lower().strip()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.lower() in command_lower:
            return False, f"Dangerous command pattern detected: '{pattern}'"
    return True, ""

def run_command(command: str, timeout: int = 60) -> str:
    """
    Execute a Shell/terminal command.
    Parameters:
        command: Command to execute
        timeout: Timeout in seconds, defaults to 60
    """
    logger.debug(f"(run_command: {command})")
    is_safe, reason = _is_command_safe(command)
    if not is_safe:
        return f"Security error: {reason}"
    
    try:
        use_shell = any(c in command for c in ['|', '>', '<', '&&', '||', ';', '*', '?'])
        
        if use_shell:
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
        else:
            if _platform.system() == "Windows":
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
            else:
                cmd_parts = shlex.split(command)
                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    cwd=str(base_dir)
                )
        
        output = result.stdout + result.stderr
        return_code = result.returncode
        return f"Return code: {return_code}\nOutput:\n{output}" if output else f"Execution completed, return code: {return_code}"
    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out ({timeout} seconds)"
    except Exception as e:
        return f"Execution error: {e}"


def edit_file(name: str, old_text: str, new_text: str) -> str:
    """
    Edit a file by replacing old_text with new_text (replaces first occurrence only).
    Parameters:
        name: File name/path
        old_text: Original text to replace
        new_text: New text to substitute
    """
    logger.debug(f"(edit_file {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"Error: File '{name}' does not exist"
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if old_text not in content:
            return f"Error: Text to replace not found"
        
        new_content = content.replace(old_text, new_text, 1)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        return f"File '{name}' edited successfully"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Edit error: {e}"


def append_file(name: str, content: str) -> str:
    """
    Append content to the end of a file.
    Parameters:
        name: File name/path
        content: Content to append
    """
    logger.debug(f"(append_file {name})")
    try:
        file_path = _safe_path(name)
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
        return f"Content appended to '{name}'"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Append error: {e}"


def search_in_files(keyword: str, file_extension: str = None) -> str:
    """
    Search for a keyword in files.
    Parameters:
        keyword: Keyword to search for
        file_extension: Optional, limit search to specific file types, e.g., ".py", ".txt"
    """
    logger.debug(f"(search_in_files keyword='{keyword}', ext={file_extension})")
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
            output = f"Found {len(results)} matches:\n" + "\n".join(results[:50])
            if len(results) > 50:
                output += f"\n... {len(results) - 50} more matches not shown"
            return output
        return "No matches found"
    except Exception as e:
        return f"Search error: {e}"


def create_directory(name: str) -> str:
    """
    Create a directory.
    Parameters:
        name: Directory name/path
    """
    logger.debug(f"(create_directory {name})")
    try:
        dir_path = _safe_path(name)
        os.makedirs(dir_path, exist_ok=True)
        return f"Directory '{name}' created successfully"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error creating directory: {e}"


def delete_directory(name: str, force: bool = False) -> str:
    """
    Delete a directory.
    Parameters:
        name: Directory name/path
        force: Whether to force delete non-empty directories
    """
    logger.debug(f"(delete_directory {name}, force={force})")
    try:
        import shutil
        dir_path = _safe_path(name)
        if not dir_path.exists():
            return f"Error: Directory '{name}' does not exist"
        if not dir_path.is_dir():
            return f"Error: '{name}' is not a directory"
        
        if force:
            shutil.rmtree(dir_path)
        else:
            os.rmdir(dir_path)  # Can only delete empty directories
        return f"Directory '{name}' has been deleted"
    except OSError as e:
        if "not empty" in str(e).lower() or "目录不是空的" in str(e):
            return f"Error: Directory is not empty, set force=True to force delete"
        return f"Delete error: {e}"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Delete error: {e}"


def http_request(url: str, method: str = "GET", data: str = None, headers: str = None) -> str:
    """
    Send an HTTP request (general API call).
    Parameters:
        url: Request URL
        method: Request method (GET, POST, PUT, DELETE, PATCH)
        data: Request body data (JSON string format)
        headers: Request headers (JSON string format)
    """
    logger.debug(f"(http_request {method} {url})")
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
        
        return f"Status code: {response.status_code}\nResponse:\n{resp_text[:8000]}{'...' if len(resp_text) > 8000 else ''}"
    except json.JSONDecodeError as e:
        return f"JSON parsing error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"Error: {e}"


def get_file_info(name: str) -> str:
    """
    Get detailed file information (size, modification time, line count, etc.).
    Parameters:
        name: File name/path
    """
    logger.debug(f"(get_file_info {name})")
    try:
        file_path = _safe_path(name)
        if not file_path.exists():
            return f"Error: File '{name}' does not exist"
        
        stat = file_path.stat()
        info = [
            f"File: {name}",
            f"Size: {stat.st_size} bytes",
            f"Modified: {datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Created: {datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
                info.append(f"Lines: {line_count}")
            except:
                pass
        
        return "\n".join(info)
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error getting file info: {e}"


def copy_file(source: str, destination: str) -> str:
    """
    Copy a file.
    Parameters:
        source: Source file path
        destination: Destination file path
    """
    logger.debug(f"(copy_file {source} -> {destination})")
    try:
        import shutil
        src_path = _safe_path(source)
        dst_path = _safe_path(destination)
        
        if not src_path.exists():
            return f"Error: Source file '{source}' does not exist"
        
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return f"File copied: '{source}' -> '{destination}'"
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Copy error: {e}"


workers_tools = [
    # 文件操作
    get_file_info,
    list_files,
    read_file,
    write_file,
    edit_file,
    append_file,
    copy_file,
    rename_file,
    delete_file,
    # 目录操作
    create_directory,
    delete_directory,
    # 搜索操作
    search_in_files,
    search_web,
    # 网络操作
    fetch_webpage,
    http_request,
    # 执行操作
    run_command,
    execute_file,
    # 多模态图像理解
    MultimodalTools.analyze_local_image,
    MultimodalTools.analyze_image_url,
    MultimodalTools.analyze_multiple_images,
    MultimodalTools.analyze_videos_url,
]

workers_parameter = {
    "temperature": 0.6,
    "top_p": 0.8,
    "max_tokens": 65536,
}
