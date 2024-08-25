import openai


# Method Explanation:
# Chain of Thought (CoT) reasoning is a process where the model is encouraged to break down
# complex problems into smaller, manageable steps, reasoning through each one step by step.
# This method helps in improving the model's ability to handle multi-step tasks by making
# the reasoning process explicit and structured. CoT is particularly effective in tasks
# that require sequential reasoning or where the final answer depends on correctly
# processing a series of intermediate steps.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-3.5-mini-instruct model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Method Explanation:
# Chain of Thought (CoT) reasoning breaks down complex problems into smaller, manageable steps.

def chain_of_thought_reasoning(task: str) -> str:
    prompt = f"Let's break this problem down step by step to find the solution:\n\nTask: {task}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# Example task
task = "If a train leaves at 2 PM traveling 60 mph, and another train leaves at 3 PM traveling 75 mph, when will they meet?"
result = chain_of_thought_reasoning(task)
print(result)
