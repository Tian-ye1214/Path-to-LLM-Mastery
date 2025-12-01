import BasicTools
import ManagementTools
import MultimodalTools
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from prompt import manager_system_prompt, workers_system_prompt
from typing import Tuple
import os
from dotenv import load_dotenv
load_dotenv()

provider = OpenAIProvider(
    base_url=os.environ.get('BASE_URL'),
    api_key=os.environ.get('API_KEY')
)


def create_model(model_name: str, parameter: dict):
    """创建模型实例"""
    return OpenAIChatModel(
        model_name,
        provider=provider,
        settings=ModelSettings(**parameter)
    )


def create_working_agent(model_name: str = "deepseek-chat", parameter: dict = None):
    """创建工作Agent - 负责执行具体任务"""
    if parameter is None:
        parameter = {
            "temperature": 0.6,
            "top_p": 0.8,
        }

    all_tools = [
        # 文件操作
        BasicTools.get_file_info,
        BasicTools.list_files,
        BasicTools.read_file,
        BasicTools.write_file,
        BasicTools.edit_file,
        BasicTools.append_file,
        BasicTools.copy_file,
        BasicTools.rename_file,
        BasicTools.delete_file,
        # 目录操作
        BasicTools.create_directory,
        BasicTools.delete_directory,
        # 搜索操作
        BasicTools.search_in_files,
        BasicTools.search_web,
        # 网络操作
        BasicTools.fetch_webpage,
        BasicTools.http_request,
        # 执行操作
        BasicTools.run_command,
        BasicTools.execute_file,
        # 多模态图像理解
        MultimodalTools.analyze_local_image,
        MultimodalTools.analyze_image_url,
        MultimodalTools.analyze_multiple_images,
        MultimodalTools.analyze_videos_url,
    ]
    
    model = create_model(model_name, parameter)
    agent = Agent(
        model,
        tools=all_tools,
        system_prompt=workers_system_prompt
    )
    return agent


def create_management_agent(model_name: str = "deepseek-reasoner", parameter: dict = None):
    """创建管理Agent - 负责任务规划和协调"""
    if parameter is None:
        parameter = {
            "temperature": 0.3,
            "top_p": 0.95,
        }

    management_tools = [
        ManagementTools.create_todo_list,
        ManagementTools.get_todo_list,
        ManagementTools.mark_task_complete,
        ManagementTools.mark_task_failed,
        ManagementTools.get_final_summary,
        ManagementTools.get_next_pending_task,
        ManagementTools.check_task_can_retry,
    ]
    
    model = create_model(model_name, parameter)
    agent = Agent(
        model,
        tools=management_tools,
        system_prompt=manager_system_prompt
    )
    return agent


def execute_task_with_worker(worker_agent: Agent, task_description: str, 
                              user_goal: str = "", retry_info: str = "", 
                              history: list = None) -> Tuple[bool, str, list]:
    """
    使用工作Agent执行任务
    返回: (是否成功, 结果/失败原因, 更新后的历史消息)
    """
    prompt = f"【用户最终目标】\n{user_goal}\n\n【当前任务】\n请执行以下任务:\n\n{task_description}"
    if retry_info:
        prompt += f"\n\n这是重试执行，之前的失败信息:\n{retry_info}\n请尝试用不同的方法完成任务。"
    
    try:
        print(f"\n{'='*50}")
        print(f"Working Agent 开始执行任务...")
        print(f"当前任务: {task_description}")
        if retry_info:
            print(f"重试信息: {retry_info}")
        print(f"{'='*50}")

        result = worker_agent.run_sync(prompt, message_history=history)
        
        output = result.output
        history = list(result.all_messages())

        print(f"\n{'='*50}")
        print(history)
        print(f"\n{'='*50}")
        
        print(f"\nWorking Agent 返回:\n{output}\n")

        if "SUCCESS" in output.upper() or "成功" in output:
            return True, output, history
        elif "FAILED" in output.upper() or "失败" in output or "错误" in output:
            return False, output, history
        else:
            return True, output, history
            
    except Exception as e:
        error_msg = f"执行异常: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg, history or []


def run_multi_agent_system(user_input: str,
                           manager_model: str = "deepseek-reasoner",
                           worker_model: str = "deepseek-chat",
                           manager_history: list = [],):
    """
    运行多Agent系统
    
    工作流程:
    1. 管理Agent分析用户请求，创建Todo List
    2. 依次执行每个任务，使用工作Agent
    3. 处理任务结果，失败则重试（最多3次）
    4. 所有任务完成后生成最终报告
    """
    worker_history = []
    manager_agent = create_management_agent(manager_model)
    worker_agent = create_working_agent(worker_model)

    print("当前步骤: 管理Agent分析任务并创建Todo List...")
    planning_prompt = f"""请分析以下用户请求，并创建详细的任务列表（Todo List）。

用户请求: {user_input}

请使用 create_todo_list 工具创建任务列表。任务应该按照执行顺序排列，并考虑任务之间的依赖关系。
每个任务的描述应该足够详细，让执行Agent能够理解并完成。
"""
    
    try:
        result = manager_agent.run_sync(planning_prompt, message_history=manager_history)
        manager_history = result.all_messages()
    except Exception as e:
        print(f"任务规划失败: {e}")
        exit()

    print("\n" + "="*60)
    print("当前步骤: 开始执行任务...")
    print("="*60 + "\n")
    
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1

        task_manager = ManagementTools.task_manager
        print(f"\n{task_manager.get_todo_list()}\n")
        next_task = task_manager.get_next_task()
        
        if next_task is None:
            if task_manager.is_all_completed():
                print("所有任务已完成！")
                break
            elif task_manager.has_failed_tasks():
                print("存在无法完成的任务")
                break
            else:
                print("没有可执行的任务")
                break

        task_manager.mark_task_in_progress(next_task.id)
        
        print(f"\n{'='*40}")
        print(f"📌 执行任务 [{next_task.id}]: {next_task.description}")
        if next_task.retry_count > 0:
            print(f"   (第 {next_task.retry_count + 1} 次尝试)")
        print(f"{'='*40}")

        retry_info = ""
        if next_task.failure_history:
            retry_info = "之前的失败记录:\n" + "\n".join([
                f"第{i+1}次: {reason}" 
                for i, reason in enumerate(next_task.failure_history)
            ])

        success, result, worker_history = execute_task_with_worker(
            worker_agent, 
            next_task.description,
            user_goal=user_input,
            retry_info=retry_info,
            history=worker_history
        )

        if success:
            task_manager.mark_task_complete(next_task.id, result)
            print(f"✅ 任务 [{next_task.id}] 完成")
        else:
            fail_result = task_manager.mark_task_failed(next_task.id, result)
            print(f"⚠️ 任务 [{next_task.id}] 失败")
            print(fail_result)

    print("\n" + "="*60)
    print("当前步骤: 生成最终报告")
    print("="*60 + "\n")
    
    final_summary = ManagementTools.task_manager.get_final_summary()
    print(final_summary)

    summary_prompt = f"""任务执行已完成。请根据以下执行报告，直接回答用户的原始问题。

用户原始问题: {user_input}

执行报告:
{final_summary}

重要提示：
- 不要报告任务执行情况（如"创建了文件"、"任务成功完成"等）
- 直接回答用户的问题，就像你是在和用户对话一样
- 从执行报告的任务结果中提取关键信息来回答用户
- 如果任务失败导致无法回答，简要说明无法获取信息的原因

例如：
- 如果用户问"温江天气如何"，你应该回复天气情况，而不是"成功查询了天气"
- 如果用户问"帮我写个脚本"，你应该告诉用户脚本已保存到哪里、主要功能是什么
"""
    
    try:
        final_result = manager_agent.run_sync(summary_prompt)
        print("\n" + "="*60)
        print("🎯 最终回复")
        print("="*60)
        print(final_result.output)
        return final_result.output, manager_history
    except Exception as e:
        return final_summary, manager_history


def main():
    """主函数 - 交互式运行"""
    print("="*60)
    print("输入 '新任务' 可以清除上下文重新开始")
    print("输入 'quit' 或 'exit' 退出程序")
    print("="*60 + "\n")
    manager_history = []

    while True:
        try:
            user_input = input("\n📝 请输入您的任务: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if '新任务' in user_input:
                manager_history = []

            result, manager_history = run_multi_agent_system(
                user_input,
                manager_model='gpt-5.1',
                worker_model='gpt-5-mini',
                manager_history=manager_history
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
