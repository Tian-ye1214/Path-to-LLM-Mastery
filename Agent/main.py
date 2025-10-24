import tools
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIChatModel(
    'deepseek-reasoner',
    provider=DeepSeekProvider(api_key=''),
)

system_prompt = """
You are an experienced programmer and need to find a way to complete the user's instructions. 
If the current question cannot be answered based on your knowledge, use "write_file Tool" to create a tool and use "execute_file Tool" to execute the task.
"""

agent = Agent(model,
              system_prompt=system_prompt,
              tools=[tools.read_file, tools.list_files, tools.rename_file, tools.execute_file,
                     tools.delete_file, tools.write_file, tools.search_web, tools.fetch_webpage])

def main():
    history = []
    while True:
        user_input = input("Input: ")
        resp = agent.run_sync(user_input,
                              message_history=history)
        history = list(resp.all_messages())
        print(resp.output)


if __name__ == "__main__":
    main()