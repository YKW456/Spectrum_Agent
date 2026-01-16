import os
import random
import torch
import csv
import json
from ToolClass import *
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from bayes_config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODELS_TO_TEST = [
    # 'qwen2.5-2',
    # 'gemma2',
    # 'qwen2.5',
    # 'mistral',
    # 'llama',
    # 'deepseek'
    # 'qwen2.5-merged_model2',
    'Qwen2.5_7b'
]





def load_model(model_name):

    print(f"\n=== loading: {model_name} ===")
    model_path = os.path.join("../Model", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).eval()


    from langchain_huggingface import HuggingFacePipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)


def add_one_history(memory, new, type):
    memory += type
    memory += str(new)
    memory += '\n'
    return memory


def run_experiment(llm, random_num, history_data):
    memory = ""
    validator = ToolCallValidator()

    session_id = "design_"
    if random_num >= 0:
        memory = add_one_history(memory, history_data['Query'], 'Query:')
    if random_num >= 1:
        memory = add_one_history(memory, history_data['Thought_1'], 'Thought:')
        memory = add_one_history(memory, history_data['Action_1'], 'Action:')
        memory = add_one_history(memory, history_data['Action Input_1'], 'Action Input:')
        memory = add_one_history(memory, history_data['Observation_1'], 'Observation:')
    if random_num == 2:
        memory = add_one_history(memory, history_data['Thought_2'], 'Thought:')
        memory = add_one_history(memory, history_data['Action_2'], 'Action:')
        memory = add_one_history(memory, history_data['Action Input_2'], 'Action Input:')
        memory = add_one_history(memory, history_data['Observation_2'], 'Observation:')

    agent = create_react_agent(llm, tools, custom_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=1
    )

    result = agent_executor.invoke({"input": memory}, {"callbacks": [validator]})
    print('input:', memory)

    is_correct = False
    print('Config.Tool_last_use=', Config.Tool_last_use)
    if random_num == 0 and validator.tool_triggered == "Search_Materials_Tool" and Config.Tool_last_use == 0:
        is_correct = True
    elif random_num == 1 and validator.tool_triggered == "Update_Config" and Config.Tool_last_use == 1:
        is_correct = True
    elif random_num == 2 and validator.tool_triggered is None:
        is_correct = True
    return is_correct


if __name__ == '__main__':
    search_tool = Search_Materials_Tool()
    update_tool = Update_Config()
    tools = [search_tool, update_tool]

    json_pth = 'model_outputs_test.json'
    with open(json_pth, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs('results', exist_ok=True)

    for model_name in MODELS_TO_TEST:
        print(f"\n=== Testing: {model_name} ===")

        Config.Total_count = 0
        Config.Correct_count = 0
        Config.Results = []


        llm = load_model(model_name)

        total_runs = len(data)

        for i in range(total_runs):
            Config.Tool_last_use = 4
            history_data = data[i]['output']
            if i < total_runs / 3:
                random_num = 0
            elif i < 2 * total_runs / 3:
                random_num = 1
            else:
                random_num = 2

            is_correct = run_experiment(llm, random_num, history_data)

            Config.Total_count += 1
            if is_correct:
                Config.Correct_count += 1

            Config.Results.append({
                'run_id': i,
                'random_num': random_num,
                'model_name': model_name,
                'is_correct': is_correct
            })

            print(f'Run {i}: random_num={random_num}, correct={is_correct}')

        total_correct = sum(1 for r in Config.Results if r['is_correct'])
        zero_correct = sum(1 for r in Config.Results if r['random_num'] == 0 and r['is_correct'])
        one_correct = sum(1 for r in Config.Results if r['random_num'] == 1 and r['is_correct'])
        two_correct = sum(1 for r in Config.Results if r['random_num'] == 2 and r['is_correct'])

        zero_total = sum(1 for r in Config.Results if r['random_num'] == 0)
        one_total = sum(1 for r in Config.Results if r['random_num'] == 1)
        two_total = sum(1 for r in Config.Results if r['random_num'] == 2)

        accuracy_total = total_correct / Config.Total_count if Config.Total_count > 0 else 0
        accuracy_zero = zero_correct / zero_total if zero_total > 0 else 0
        accuracy_one = one_correct / one_total if one_total > 0 else 0
        accuracy_two = two_correct / two_total if two_total > 0 else 0

        csv_filename = f'results/{model_name}_tool_results.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model_name', 'random_num_0_accuracy', 'random_num_1_accuracy',
                          'random_num_2_accuracy', 'total_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({
                'model_name': model_name,
                'random_num_0_accuracy': accuracy_zero,
                'random_num_1_accuracy': accuracy_one,
                'random_num_2_accuracy': accuracy_two,
                'total_accuracy': accuracy_total
            })

        print("\n=== Statistics ===")
        print(f"Model: {model_name}")
        print(f"random_num=0 accuracy: {accuracy_zero:.2f}")
        print(f"random_num=1 accuracy: {accuracy_one:.2f}")
        print(f"random_num=2 accuracy: {accuracy_two:.2f}")
        print(f"Total accuracy: {accuracy_total:.2f}")
        print(f"result save to {csv_filename}")
        del llm
        torch.cuda.empty_cache()
