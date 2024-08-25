import openai



# Method Explanation:
# Commonsense reasoning enables models to make inferences based on everyday knowledge that
# people typically take for granted. This type of reasoning is crucial for understanding
# and interacting with the world in a way that aligns with human expectations.
# Datasets like ART (Abductive Natural Language Inference) help assess how well models
# can perform commonsense reasoning by requiring them to infer plausible explanations
# for given situations.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-3.5-mini-instruct model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Method Explanation:
# Commonsense reasoning enables models to make inferences based on everyday knowledge.

def commonsense_reasoning(scenario: str) -> str:
    prompt = f"Given the following scenario:\n\n{scenario}\n\nWhat is the most plausible explanation or conclusion?"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# Example scenario
scenario = "John left the house with an umbrella even though it wasn't raining."
result = commonsense_reasoning(scenario)
print(result)
