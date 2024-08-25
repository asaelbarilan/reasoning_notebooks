import openai
import numpy as np
from key import get_personal_key
# Method Explanation:
# Self-consistency reasoning is an approach that improves the robustness of model outputs
# by generating multiple reasoning paths for the same problem and selecting the most consistent one.
# This method is particularly useful in tasks where different interpretations or reasoning paths
# might lead to different conclusions. By averaging across these multiple paths, the model
# can arrive at a more reliable and accurate conclusion.

# Set up your OpenAI API key
openai.api_key = get_personal_key()

def self_consistency_reasoning(task: str, num_samples: int = 5) -> str:
    responses = []
    for _ in range(num_samples):
        prompt = f"Consider this problem and provide a thoughtful answer:\n\nTask: {task}"
        response = openai.ChatCompletion.create(
            model='gpt-4.0',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=150,
            temperature=0.9,
        )
        responses.append(response.choices[0].message['content'].strip())

    final_answer = max(set(responses), key=responses.count)
    return final_answer

# Example task
task = "What's the next number in the sequence: 2, 4, 8, 16, ...?"
result = self_consistency_reasoning(task)
print(result)
