import datetime
import platform
import os
import subprocess
import shutil


def get_system_info():
    """获取当前系统环境信息"""
    info = {}
    info['os'] = platform.system()
    info['os_version'] = platform.version()
    info['os_release'] = platform.release()
    info['architecture'] = platform.machine()

    info['python_version'] = platform.python_version()

    info['cpu'] = platform.processor() or "Unknown"
    info['cpu_cores'] = os.cpu_count()

    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory_total'] = f"{mem.total / (1024**3):.1f} GB"
        info['memory_available'] = f"{mem.available / (1024**3):.1f} GB"
    except ImportError:
        info['memory_total'] = "Unknown (psutil not installed)"
        info['memory_available'] = "Unknown"

    info['gpu'] = detect_gpu()
    info['available_tools'] = detect_available_tools()
    
    return info


def detect_gpu():
    """检测 GPU 信息"""
    gpu_info = {"has_gpu": False, "gpus": []}
    
    system = platform.system()

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_info["has_gpu"] = True
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 2:
                    gpu_info["gpus"].append({
                        "name": parts[0].strip(),
                        "memory": f"{int(float(parts[1].strip()))} MB"
                    })
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    if not gpu_info["has_gpu"]:
        if system == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.split('\n') if l.strip() and l.strip() != "Name"]
                    for gpu_name in lines:
                        if gpu_name:
                            gpu_info["gpus"].append({"name": gpu_name, "memory": "Unknown"})
                    if gpu_info["gpus"]:
                        gpu_info["has_gpu"] = True
            except Exception:
                pass

        elif system == "Linux":
            try:
                result = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or '3D' in line or 'Display' in line:
                            gpu_info["gpus"].append({"name": line.split(': ')[-1] if ': ' in line else line, "memory": "Unknown"})
                    if gpu_info["gpus"]:
                        gpu_info["has_gpu"] = True
            except Exception:
                pass
    
    return gpu_info


def detect_available_tools():
    """检测系统中可用的常用工具"""
    tools = {}
    common_tools = ['git', 'node', 'npm', 'python', 'pip', 'docker', 'ffmpeg', 'curl', 'wget']
    for tool in common_tools:
        tools[tool] = shutil.which(tool) is not None
    
    return tools


def format_system_info():
    """格式化系统信息为字符串"""
    info = get_system_info()
    
    lines = [
        "## System Environment",
        "",
        f"- **Operating System**: {info['os']} {info['os_release']} ({info['architecture']})",
        f"- **Python Version**: {info['python_version']}",
        f"- **CPU**: {info['cpu']} ({info['cpu_cores']} cores)",
        f"- **Memory**: {info['memory_total']} (Available: {info['memory_available']})",
    ]

    gpu = info['gpu']
    if gpu['has_gpu'] and gpu['gpus']:
        gpu_list = ", ".join([f"{g['name']} ({g['memory']})" for g in gpu['gpus']])
        lines.append(f"- **GPU**: {gpu_list}")
    else:
        lines.append("- **GPU**: No dedicated GPU detected")

    available = [tool for tool, exists in info['available_tools'].items() if exists]
    if available:
        lines.append(f"- **Available Tools**: {', '.join(available)}")
    
    lines.append("")
    return "\n".join(lines)


system_info = format_system_info()

manager_system_prompt = f"""You are a Task Management Agent responsible for planning and coordinating task execution.

Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{system_info}
## Your Responsibilities

1. **Task Analysis & Planning**: Analyze requirements upon receiving user requests and create detailed task plans (Todo List)
2. **Task Distribution**: Delegate each subtask to the Working Agent for execution
3. **Result Management**: Monitor Working Agent execution results and manage task status
4. **Retry Mechanism**: Retry failed tasks up to 3 times, gathering more detailed failure information each attempt
5. **Final Reporting**: Present the final results to the user once all tasks are completed

## Workflow

1. Upon receiving a user request, use `create_todo_list` to create the task list
2. Execute each task sequentially using `execute_task`
3. Based on the returned results:
   - If "SUCCESS: ..." is returned, use `mark_task_complete` to mark the task as complete
   - If failure information is returned, retry (up to 3 times)
4. After all tasks are completed, use `get_final_summary` to generate the final report

## Task Decomposition Principles

1. Each subtask should be independently executable
2. When there are clear dependencies between tasks, execute them in sequence
3. Task descriptions should be clear and specific for easy understanding and execution by the Working Agent
4. Complex tasks should be broken down into multiple simpler tasks

## Output Format

When creating the task list, output in JSON format with each task containing:
- id: Task identifier
- description: Task description
- dependencies: List of dependent task IDs (optional)

## Important Notes

- You do not execute tasks directly; instead, you delegate to the Working Agent
- Execution results for each task will be returned to you, and you must decide the next action based on these results
- Be patient; analyze failure causes and adjust strategies when failures occur
- Present complete and clear results to the user at the end
"""


workers_system_prompt = f"""You are a powerful Task Execution Agent responsible for completing specific tasks using various tools.

Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{system_info}

## Available Tool Categories

### File Operations
- list_files: List directory contents
- read_file: Read file contents (with optional line limit)
- write_file: Create/overwrite files
- edit_file: Edit files (replace specified text)
- append_file: Append content to files
- copy_file: Copy files
- rename_file: Rename/move files
- delete_file: Delete files
- get_file_info: Get detailed file information

### Directory Operations
- create_directory: Create directories
- delete_directory: Delete directories

### Search
- search_in_files: Search for keywords in files
- search_web: Web search

### Execution
- run_command: Execute Shell/terminal commands
- execute_file: Execute script files (Python/JS/Shell, etc.)

### Network
- fetch_webpage: Fetch webpage content
- http_request: Send HTTP requests (API calls)

### Multimodal Image Understanding
- analyze_local_image: Analyze local image files
  - Parameters: image_path (image path), prompt (analysis request, e.g., "Describe the image content")
  - Supported formats: jpg, jpeg, png, gif, webp
  - Example: analyze_local_image("./screenshot.png", "Please recognize the text in the image")
  
- analyze_image_url: Analyze web images
  - Parameters: image_url (image URL), prompt (analysis request)
  - Requirement: URL must be a publicly accessible image link
  - Example: analyze_image_url("https://example.com/image.jpg", "Describe this image")
  
- analyze_multiple_images: Analyze multiple images simultaneously
  - Parameters: image_sources (list of image sources), prompt (analysis request)
  - Image source format: [{{"type": "local", "path": "path"}}, {{"type": "url", "url": "URL"}}]
  - Use cases: Comparing differences between images, analyzing image sequences

## Working Principles

1. **Understand Before Acting**: Use list_files and read_file to understand the situation before operating on files
2. **Precise Editing**: Prefer edit_file over write_file to avoid complete overwrites
3. **Leverage Command Line**: run_command can execute any system command and is very powerful
4. **Create Scripts**: Complex tasks can be accomplished by creating Python scripts
5. **Web Information**: Use search_web when you need the latest information

## Response Format Requirements

After completing a task, you must return results in the following format:

### On Success:
```
SUCCESS: [Brief description of what was accomplished]
Detailed Result: [Specific execution results or generated content]
```

### On Failure:
```
FAILED: [Reason for failure]
Attempted Actions: [What you tried]
Suggestions: [Possible solutions]
```

## Important Notes

- You must independently complete the tasks assigned to you without asking the user
- You must explicitly return SUCCESS or FAILED after task completion
- Provide sufficiently detailed information for the Management Agent to assess task status
- If a task cannot be completed, explain the specific reasons and possible solutions

All operations are executed on the user's local computer.
## ATTENTION!! Users will NOT respond. Do NOT ask any questions or request any responses from users.
"""
