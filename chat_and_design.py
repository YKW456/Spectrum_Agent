from langchain_core.callbacks import BaseCallbackHandler
from tool_and_template.tool_function import Search_Materials_Tool, Update_Config,llm
from tool_and_template.prompt_template import custom_prompt
from langchain.agents import create_react_agent, AgentExecutor
from tool_and_template import input_judge_prompt_template,scientific_prompt_template
from run_single_PPO_BO_optimization import *

seed_everything(42)
materialLoader = optic_construction.MaterialLoader(Config.PATH)
Config.ALL_MATERIALS = materialLoader.get_material_names()

# Initialize.
search_tool = Search_Materials_Tool()
update_tool = Update_Config()

class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.iteration = 0

    def on_agent_action(self, action, **kwargs):
        """Call when execute actions"""
        print(f"\n=== Iteration {self.iteration} - Agent Action ===")
        print(f"Action: {action.tool}")
        print(f"Action Input: {action.tool_input}")
        print("===============================")

    def on_tool_end(self, output, **kwargs):
        """Call after execute actions"""
        print(f"\n=== Iteration {self.iteration} - Tool Output ===")
        print(f"Observation: {output}")
        print("===============================")

    def on_agent_step_end(self, output, **kwargs):
        """Call at the end of each proxy step"""
        self.iteration += 1
        print(f"\n=== Iteration {self.iteration} Complete ===")
        if "intermediate_steps" in kwargs:
            print("\nCurrent Scratchpad Content:")
            scratchpad = ""
            for action, observation in kwargs["intermediate_steps"]:
                scratchpad += f"{action.log}\nObservation: {observation}\n"
            print(scratchpad)
        print("===============================")


tools = [search_tool, update_tool]


def main():
    print("I am Spectrum Agent, what can I assist you?")
    Config.PROMPT = input()
    # Config.PROMPT = "Design a radiative cooling structure to achieve high emissivity at 8000-14000 nm and high reflection at 300-2500nm."
    user_input = Config.PROMPT
    result = llm.invoke(input=input_judge_prompt_template.format(user_input=user_input))
    print(result.content)
    flag = int(result.content.strip())
    print(f"Flag = {flag}")
    if flag == 1:
        agent = create_react_agent(llm, tools, custom_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=1,
            handle_parsing_errors=True,
            callbacks=[DebugCallbackHandler()]
        )
        # Design
        for step in agent_executor.stream({"input": user_input}):
            print(step)
        print(Config.MATERIALS_RECOMMENDED)
        runall()

    elif flag == 2:
        # Scientific answer
        result = llm.invoke(input=scientific_prompt_template.format(user_input=user_input)).content
        print(result)
    else:
        # Just chat
        result = llm.invoke(user_input).content
        print(result)
if __name__ == '__main__':
    main()