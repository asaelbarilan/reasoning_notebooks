import openai
from key import get_personal_key

# Method Explanation:
# Chain of Thought (CoT) reasoning is a process where the model is encouraged to break down
# complex problems into smaller, manageable steps, reasoning through each one step by step.
# This method helps in improving the model's ability to handle multi-step tasks by making
# the reasoning process explicit and structured. CoT is particularly effective in tasks
# that require sequential reasoning or where the final answer depends on correctly
# processing a series of intermediate steps.

# Set up your OpenAI API key
openai.api_key = get_personal_key()


def chain_of_thought_reasoning(task: str) -> str:
    prompt = f"Let's break this problem down step by step to find the solution:\n\nTask: {task}"
    response = openai.ChatCompletion.create(
        model='gpt-4.0',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

# Example task
task = "If a train leaves at 2 PM traveling 60 mph, and another train leaves at 3 PM traveling 75 mph, when will they meet?"
result = chain_of_thought_reasoning(task)
print(result)
