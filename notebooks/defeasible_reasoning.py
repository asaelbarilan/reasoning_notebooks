 # Method Explanation:
# Defeasible reasoning is a form of reasoning where the conclusions drawn can be retracted
# if new information contradicts them. It allows models to dynamically adjust their conclusions
# based on additional evidence, making it particularly useful in real-world scenarios where
# not all information is initially available. This type of reasoning is explored in NLP using
# datasets that require models to revise their conclusions as new information is introduced.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-3.5-mini-instruct model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Method Explanation:
# Defeasible reasoning adjusts conclusions based on additional evidence.

def defeasible_reasoning(premises: str, new_information: str) -> str:
 prompt = f"Given the following premises:\n\n{premises}\n\nNow, consider this new information:\n\n{new_information}\n\nWhat conclusions can you draw now?"

 inputs = tokenizer(prompt, return_tensors="pt")
 outputs = model.generate(**inputs, max_new_tokens=150)
 response = tokenizer.decode(outputs[0], skip_special_tokens=True)

 return response


# Example premises and new information
premises = "1. It is cloudy outside.\n2. When it is cloudy, it usually rains."
new_information = "However, the weather forecast says it will be sunny."
result = defeasible_reasoning(premises, new_information)
print(result)


