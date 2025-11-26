from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import json_repair as json


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """任务数据结构"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    failure_history: List[str] = field(default_factory=list)


class TaskManager:
    """任务管理器 - 管理Todo List"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []
    
    def create_todo_list(self, tasks_json: str) -> str:
        """
        根据JSON创建任务列表。
        Parameters:
            tasks_json: JSON格式的任务列表，格式为:
                [{"id": "1", "description": "任务描述", "dependencies": ["依赖任务id"]}]
        """
        print(f"(create_todo_list)")
        try:
            tasks_data = json.loads(tasks_json)
            self.tasks.clear()
            self.task_order.clear()
            
            for task_data in tasks_data:
                task_id = str(task_data.get("id", len(self.tasks) + 1))
                task = Task(
                    id=task_id,
                    description=task_data.get("description", ""),
                    dependencies=task_data.get("dependencies", [])
                )
                self.tasks[task_id] = task
                self.task_order.append(task_id)
            
            return self._format_todo_list()
        except json.JSONDecodeError as e:
            return f"错误: JSON解析失败 - {e}"
        except Exception as e:
            return f"错误: 创建任务列表失败 - {e}"
    
    def _format_todo_list(self) -> str:
        """格式化输出Todo List"""
        if not self.tasks:
            return "任务列表为空"
        
        lines = ["任务列表 (Todo List)", "=" * 40]
        for task_id in self.task_order:
            task = self.tasks[task_id]
            status_icon = {
                TaskStatus.PENDING: "⬜",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌"
            }.get(task.status, "⬜")
            
            line = f"{status_icon} [{task.id}] {task.description}"
            if task.dependencies:
                line += f" (依赖: {', '.join(task.dependencies)})"
            if task.retry_count > 0:
                line += f" [重试: {task.retry_count}/{task.max_retries}]"
            lines.append(line)

        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        total = len(self.tasks)
        lines.append("=" * 40)
        lines.append(f"进度: {completed}/{total} ({completed/total*100:.1f}%)" if total > 0 else "进度: 0/0")
        todo_list = "\n".join(lines)
        print("=" * 60)
        print("TODO LIST:\n" + todo_list)
        print("=" * 60)
        
        return todo_list
    
    def get_next_task(self) -> Optional[Task]:
        """获取下一个可执行的任务"""
        for task_id in self.task_order:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                deps_completed = all(
                    self.tasks.get(dep_id, Task(id="", description="")).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_completed:
                    return task
        return None
    
    def mark_task_in_progress(self, task_id: str) -> str:
        """标记任务为执行中"""
        if task_id not in self.tasks:
            return f"错误: 任务 {task_id} 不存在"
        self.tasks[task_id].status = TaskStatus.IN_PROGRESS
        return f"任务 {task_id} 已开始执行"
    
    def mark_task_complete(self, task_id: str, result: str = "") -> str:
        """
        标记任务为已完成。
        Parameters:
            task_id: 任务ID
            result: 任务执行结果
        """
        print(f"(mark_task_complete {task_id})")
        if task_id not in self.tasks:
            return f"错误: 任务 {task_id} 不存在"
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        
        return f"✅ 任务 [{task_id}] 已完成\n{self._format_todo_list()}"
    
    def mark_task_failed(self, task_id: str, reason: str) -> str:
        """
        记录任务失败并增加重试次数。
        Parameters:
            task_id: 任务ID
            reason: 失败原因
        """
        print(f"(mark_task_failed {task_id})")
        if task_id not in self.tasks:
            return f"错误: 任务 {task_id} 不存在"
        
        task = self.tasks[task_id]
        task.failure_history.append(reason)
        task.retry_count += 1
        
        if task.retry_count >= task.max_retries:
            task.status = TaskStatus.FAILED
            return f"❌ 任务 [{task_id}] 已达到最大重试次数 ({task.max_retries}次)\n失败历史:\n" + \
                   "\n".join([f"  第{i+1}次: {r}" for i, r in enumerate(task.failure_history)])
        else:
            task.status = TaskStatus.PENDING  # 重置为待执行，等待重试
            return f"⚠️ 任务 [{task_id}] 执行失败，准备第 {task.retry_count + 1} 次重试\n" + \
                   f"失败原因: {reason}\n" + \
                   f"剩余重试次数: {task.max_retries - task.retry_count}"
    
    def can_retry(self, task_id: str) -> bool:
        """检查任务是否还可以重试"""
        if task_id not in self.tasks:
            return False
        task = self.tasks[task_id]
        return task.retry_count < task.max_retries
    
    def get_task_status(self, task_id: str) -> str:
        """获取任务状态"""
        if task_id not in self.tasks:
            return f"错误: 任务 {task_id} 不存在"
        task = self.tasks[task_id]
        return f"任务 [{task_id}]: {task.status.value}\n描述: {task.description}\n结果: {task.result or '无'}"
    
    def get_todo_list(self) -> str:
        """获取当前Todo List状态"""
        print("(get_todo_list)")
        return self._format_todo_list()
    
    def is_all_completed(self) -> bool:
        """检查是否所有任务都已完成"""
        return all(
            task.status == TaskStatus.COMPLETED 
            for task in self.tasks.values()
        )
    
    def has_failed_tasks(self) -> bool:
        """检查是否有失败的任务"""
        return any(
            task.status == TaskStatus.FAILED 
            for task in self.tasks.values()
        )
    
    def get_final_summary(self) -> str:
        """
        生成最终任务执行总结报告。
        """
        print("(get_final_summary)")
        lines = [
            "=" * 50,
            "📊 任务执行总结报告",
            "=" * 50,
            ""
        ]
        
        completed_tasks = []
        failed_tasks = []
        
        for task_id in self.task_order:
            task = self.tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                completed_tasks.append(task)
            elif task.status == TaskStatus.FAILED:
                failed_tasks.append(task)

        lines.append(f"✅ 已完成任务: {len(completed_tasks)}/{len(self.tasks)}")
        lines.append("-" * 40)
        for task in completed_tasks:
            lines.append(f"  [{task.id}] {task.description}")
            if task.result:
                # 缩进结果显示
                result_lines = task.result.split('\n')
                for rl in result_lines[:5]:  # 最多显示5行结果
                    lines.append(f"      → {rl}")
                if len(result_lines) > 5:
                    lines.append(f"      ... (还有 {len(result_lines) - 5} 行)")

        if failed_tasks:
            lines.append("")
            lines.append(f"❌ 失败任务: {len(failed_tasks)}")
            lines.append("-" * 40)
            for task in failed_tasks:
                lines.append(f"  [{task.id}] {task.description}")
                lines.append(f"      重试次数: {task.retry_count}")
                if task.failure_history:
                    lines.append(f"      最后失败原因: {task.failure_history[-1]}")
        
        lines.append("")
        lines.append("=" * 50)

        if self.is_all_completed():
            lines.append("所有任务已成功完成！")
        elif self.has_failed_tasks():
            lines.append("⚠部分任务执行失败，请查看失败原因。")
        else:
            lines.append("任务执行中...")
        
        return "\n".join(lines)


task_manager = TaskManager()


def create_todo_list(tasks_json: str) -> str:
    """
    创建任务列表（Todo List）。
    Parameters:
        tasks_json: JSON格式的任务列表，格式为:
            [{"id": "1", "description": "任务描述", "dependencies": []}]
    示例:
        create_todo_list('[{"id": "1", "description": "搜索相关信息"}, {"id": "2", "description": "下载文件", "dependencies": ["1"]}]')
    """
    return task_manager.create_todo_list(tasks_json)


def get_todo_list() -> str:
    """
    获取当前任务列表状态。
    """
    return task_manager.get_todo_list()


def mark_task_complete(task_id: str, result: str) -> str:
    """
    标记任务已完成。
    Parameters:
        task_id: 任务ID
        result: 任务执行结果描述
    """
    return task_manager.mark_task_complete(task_id, result)


def mark_task_failed(task_id: str, reason: str) -> str:
    """
    标记任务失败并记录原因。会自动增加重试计数。
    Parameters:
        task_id: 任务ID
        reason: 失败原因
    """
    return task_manager.mark_task_failed(task_id, reason)


def get_final_summary() -> str:
    """
    获取最终任务执行总结报告。
    在所有任务执行完毕后调用。
    """
    return task_manager.get_final_summary()


def get_next_pending_task() -> str:
    """
    获取下一个待执行的任务。
    会自动考虑任务依赖关系。
    """
    print("(get_next_pending_task)")
    task = task_manager.get_next_task()
    if task:
        task_manager.mark_task_in_progress(task.id)
        return f"📌 下一个任务:\nID: {task.id}\n描述: {task.description}\n" + \
               (f"当前重试次数: {task.retry_count}/{task.max_retries}" if task.retry_count > 0 else "")
    else:
        if task_manager.is_all_completed():
            return "✅ 所有任务已完成！"
        elif task_manager.has_failed_tasks():
            return "❌ 存在无法完成的任务，请查看失败详情。"
        else:
            return "⏳ 当前没有可执行的任务（可能在等待依赖任务完成）"


def check_task_can_retry(task_id: str) -> str:
    """
    检查任务是否还可以重试。
    Parameters:
        task_id: 任务ID
    """
    can_retry = task_manager.can_retry(task_id)
    task = task_manager.tasks.get(task_id)
    if task:
        return f"任务 [{task_id}] {'可以重试' if can_retry else '已达到最大重试次数'}\n" + \
               f"当前重试次数: {task.retry_count}/{task.max_retries}"
    return f"错误: 任务 {task_id} 不存在"

