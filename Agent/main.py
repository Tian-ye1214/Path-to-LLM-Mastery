from pydantic_ai import Agent
from prompt import manager_system_prompt, workers_system_prompt
from BasicTools import workers_tools, workers_parameter
from ManagementTools import manager_tools, manager_parameter, task_manager
from typing import Tuple
from BasicFunction import create_agent
import logger
import traceback


def execute_task_with_worker(worker_agent: Agent, task_description: str,
                             user_goal: str = "", retry_info: str = "",
                             history: list = None) -> Tuple[bool, str, list]:
    """
    使用工作Agent执行任务
    返回: (是否成功, 结果/失败原因, 更新后的历史消息)
    """
    prompt = f"[User's Ultimate Goal]\n{user_goal}\n\n[Current Task]\nPlease execute the following task:\n\n{task_description}"
    if retry_info:
        prompt += f"\n\nThis is a retry attempt. Previous failure details:\n{retry_info}\nPlease try an alternative approach to complete the task."

    try:
        logger.info("=" * 50)
        logger.info("Working Agent 开始执行任务...")
        logger.info(f"当前任务: {task_description}")
        if retry_info:
            logger.info(f"重试信息: {retry_info}")
        logger.info("=" * 50)

        result = worker_agent.run_sync(prompt, message_history=history)

        output = result.output
        history = list(result.all_messages())

        logger.info(f"Working Agent 返回:\n{output}")

        output_upper = output.upper().strip()
        output_lines = output.strip().split('\n')
        first_line = output_lines[0].upper() if output_lines else ""

        if first_line.startswith("FAILED:") or first_line.startswith("FAILED："):
            return False, output, history
        elif first_line.startswith("SUCCESS:") or first_line.startswith("SUCCESS："):
            return True, output, history
        elif output_upper.startswith("ERROR:") or output_upper.startswith("错误:") or "执行异常" in output:
            return False, output, history
        else:
            return True, output, history

    except Exception as e:
        error_msg = f"执行异常: {str(e)}"
        logger.error(f"❌ {error_msg}")

        logger.error(f"异常类型: {type(e).__name__}")
        logger.error(f"异常详情:\n{traceback.format_exc()}")

        if e.__cause__:
            logger.error(f"原始异常 (cause): {type(e.__cause__).__name__}: {e.__cause__}")
        if e.__context__ and e.__context__ != e.__cause__:
            logger.error(f"上下文异常 (context): {type(e.__context__).__name__}: {e.__context__}")
        if hasattr(e, 'args') and e.args:
            logger.error(f"异常参数: {e.args}")
        
        return False, error_msg, history or []


def run_multi_agent_system(user_input: str,
                           manager_model: str = "deepseek-reasoner",
                           worker_model: str = "deepseek-chat",
                           manager_history: list = [], ):
    """
    运行多Agent系统

    工作流程:
    1. 管理Agent分析用户请求，创建Todo List
    2. 依次执行每个任务，使用工作Agent
    3. 处理任务结果，失败则重试（最多3次）
    4. 所有任务完成后生成最终报告
    """
    task_name = user_input[:30].replace(" ", "_")
    logger.setup_task_logger(task_name)
    
    worker_history = []
    manager_agent = create_agent(manager_model, manager_parameter, manager_tools, manager_system_prompt)
    worker_agent = create_agent(worker_model, workers_parameter, workers_tools, workers_system_prompt)

    logger.info("📌 当前步骤: 创建todo list")
    planning_prompt = f"""Please analyze the following user request and create a detailed task list (Todo List).

User Request: {user_input}

Use the create_todo_list tool to generate the task list. Tasks should be arranged in execution order, with dependencies taken into consideration.
Each task description should be sufficiently detailed to enable the Worker Agent to understand and complete it.
"""

    try:
        result = manager_agent.run_sync(planning_prompt, message_history=manager_history)
        manager_history = list(result.all_messages())
    except Exception as e:
        logger.error(f"任务规划失败: {e}")
        exit()

    logger.info("=" * 60)
    logger.info("当前步骤: 开始执行任务...")
    logger.info("=" * 60)

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n{task_manager.get_todo_list()}\n")
        next_task = task_manager.get_next_task()

        if next_task is None:
            if task_manager.is_all_completed():
                logger.info("所有任务已完成！")
                break
            elif task_manager.has_failed_tasks():
                logger.warning("存在无法完成的任务")
                break
            else:
                logger.warning("没有可执行的任务")
                break

        task_manager.mark_task_in_progress(next_task.id)

        logger.info("=" * 40)
        logger.info(f"📌 执行任务 [{next_task.id}]: {next_task.description}")
        if next_task.retry_count > 0:
            logger.info(f"   (第 {next_task.retry_count + 1} 次尝试)")
        logger.info("=" * 40)

        retry_info = ""
        if next_task.failure_history:
            retry_info = "之前的失败记录:\n" + "\n".join([
                f"第{i + 1}次: {reason}"
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
            logger.info(f"✅ 任务 [{next_task.id}] 完成")
        else:
            fail_result = task_manager.mark_task_failed(next_task.id, result)
            logger.warning(f"⚠️ 任务 [{next_task.id}] 失败")
            logger.warning(fail_result)

    logger.info("")
    logger.info("=" * 60)
    logger.info("当前步骤: 生成最终报告")
    logger.info("=" * 60)

    final_summary = task_manager.get_final_summary()
    logger.info(final_summary)

    summary_prompt = f"""Task execution completed. Please respond directly to the user's original question based on the execution report below.

User's Original Question: {user_input}

Execution Report:
{final_summary}

Important Guidelines:
- Do not report task execution status (e.g., "file created", "task completed successfully")
- Respond directly to the user's question as if you were having a conversation
- Extract key information from the task results in the execution report to answer the user
- If task failures prevent a proper answer, briefly explain why the information could not be obtained

Examples:
- If the user asks "What's the weather like in Wenjiang?", respond with the weather conditions, not "Successfully queried the weather"
- If the user asks "Write me a script", tell them where the script was saved and what its main functions are
"""

    try:
        final_result = manager_agent.run_sync(summary_prompt)
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎯 最终回复")
        logger.info("=" * 60)
        logger.info(final_result.output)
        return final_result.output, manager_history
    except Exception as e:
        return final_summary, manager_history


def main():
    """主函数 - 交互式运行"""
    log = logger.get_logger()
    log.info("=" * 60)
    log.info("输入 '新任务' 可以清除上下文重新开始")
    log.info("输入 'quit' 或 'exit' 退出程序")
    log.info("=" * 60)
    manager_history = []

    while True:
        try:
            user_input = input("\n📝 请输入您的任务: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', '退出']:
                log.info("👋 再见！")
                break

            if '新任务' in user_input:
                manager_history = []
                task_manager.reset()

            result, manager_history = run_multi_agent_system(
                user_input,
                manager_model='qwen3-235b-a22b',
                worker_model='deepseek-chat',
                manager_history=manager_history
            )

        except KeyboardInterrupt:
            log.info("\n\n👋 程序已中断，再见！")
            break
        except Exception as e:
            log.error(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
