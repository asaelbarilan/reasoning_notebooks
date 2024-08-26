import openai
import numpy as np

# Method Explanation:
# Self-consistency reasoning is an approach that improves the
# robustness of model outputs
# by generating multiple reasoning paths for the same
# problem and selecting the most consistent one.
# This method is particularly useful in tasks where
# different interpretations or reasoning paths
# might lead to different conclusions. By averaging across
# these multiple paths, the model
# can arrive at a more reliable and accurate conclusion.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-3.5-mini-instruct model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Method Explanation:
# Self-consistency reasoning generates multiple reasoning paths and selects the most consistent one.

def self_consistency_reasoning(task: str, num_samples: int = 5) -> str:
    responses = []
    for _ in range(num_samples):
        prompt = f"Consider this problem and provide a thoughtful answer:\n\nTask: {task}"

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        responses.append(response)

    final_answer = max(set(responses), key=responses.count)
    return final_answer


# Example task
task = "What's the next number in the sequence: 2, 4, 8, 16, ...?"
result = self_consistency_reasoning(task)
print(result)
