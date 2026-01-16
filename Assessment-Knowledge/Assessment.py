import torch

torch.cuda.empty_cache()
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from datetime import datetime

# 配置
model_name = 'qwen-sft-Nov'
model_pth = "../Model/Qwen2.5_7b"
DATA_FILE = 'all_qa_pairs_merged_test.json'
results_dir = Path(f"model_responses_{model_name}")
results_dir.mkdir(exist_ok=True)
RESPONSES_FILE = results_dir / 'model_responses.json'
CHECKPOINT_FILE = results_dir / 'checkpoint.json'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8  


print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_pth)
model = AutoModelForCausalLM.from_pretrained(
    model_pth,
    torch_dtype=torch.bfloat16,
    device_map=device
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    # do_sample=True,
    # temperature=0.7,
    # top_p=0.85,
    # repetition_penalty=1.1,
    batch_size=batch_size  
)


def generate_batch_responses(prompts):
    
    try:
        formatted_prompts = [
            f'''You are an expert in the field of optical materials. Please answer the following questions, and your answers must meet the following requirements:
            1. Accuracy: Strictly based on scientific principles or recognized research conclusions, avoiding subjective speculation.
            2. Structured: Elaborate on core concepts (such as definitions, mechanisms, and applications) in order, requiring clear logic.
            3. The length of the generated reply should be approximately 200 words.
            Instruction:{p}'''
            for p in prompts
        ]
        responses = llm(
            formatted_prompts,
            max_length=300,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        return [r[0]['generated_text'].replace(p, "").strip()
                for p, r in zip(formatted_prompts, responses)]
    except Exception as e:
        print(f"error: {e}")
        return [None] * len(prompts)


def load_checkpoint():

    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"error: {e}")
    return {"last_index": -1, "responses": []}


def save_checkpoint(index, responses):

    checkpoint = {
        "last_index": index,
        "responses": responses,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"error: {e}")


def main():
    try:
        # load
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)


        checkpoint = load_checkpoint()
        start_index = checkpoint["last_index"] + 1
        responses = checkpoint["responses"]


        batch_questions = []
        batch_references = []
        batch_indices = []

        for i in tqdm(range(start_index, len(data)), initial=start_index, total=len(data)):
            item = data[i]
            batch_questions.append(item['Instruction'])
            batch_references.append(item['Output'])
            batch_indices.append(i)

            if len(batch_questions) == batch_size or i == len(data) - 1:
                model_responses = generate_batch_responses(batch_questions)

                for q, ref, resp, idx in zip(batch_questions, batch_references, model_responses, batch_indices):
                    responses.append({
                        "question": q,
                        "reference_answer": ref,
                        "model_answer": resp,
                    })


                if (i + 1) % (5 * batch_size) == 0 or i == len(data) - 1:
                    save_checkpoint(i, responses)
                    with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
                        json.dump(responses, f, ensure_ascii=False, indent=2)


                batch_questions = []
                batch_references = []
                batch_indices = []

        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)


        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()


    except KeyboardInterrupt:

        save_checkpoint(i, responses)

    except Exception as e:
        print(f"error: {e}")
        raise


if __name__ == "__main__":
    main()
