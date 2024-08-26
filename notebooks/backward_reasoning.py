import openai


# Method Explanation:
# Backward reasoning, also known as backward chaining, involves starting with a conclusion
# and working backward to determine the premises that would logically lead to that conclusion.
# This method is commonly used in problem-solving and decision-making tasks.
# In NLP, backward reasoning allows a model to justify conclusions by identifying
# the necessary premises, offering a more explainable approach to reasoning.


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the LLaMA model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Method Explanation:
# Backward reasoning, also known as backward chaining, involves starting with a conclusion
# and working backward to determine the premises that would logically lead to that conclusion.

def backward_reasoning(conclusion: str) -> str:
    prompt = f"Given the following conclusion:\n\n{conclusion}\n\nWhat premises would lead to this conclusion?"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# Example conclusion
conclusion = "The cat is wet."
result = backward_reasoning(conclusion)
print(result)
print('')