import openai
from key import get_personal_key

# Method Explanation:
# Backward reasoning, also known as backward chaining, involves starting with a conclusion
# and working backward to determine the premises that would logically lead to that conclusion.
# This method is commonly used in problem-solving and decision-making tasks.
# In NLP, backward reasoning allows a model to justify conclusions by identifying
# the necessary premises, offering a more explainable approach to reasoning.


# Set up your OpenAI API key
openai.api_key = get_personal_key()


def backward_reasoning(conclusion: str) -> str:
    prompt = f"Given the following conclusion:\n\n{conclusion}\n\nWhat premises would lead to this conclusion?"
    response = openai.ChatCompletion.create(
        model='gpt-4.0',
        messages=[
            {'role': 'system', 'content': 'You are a backward reasoner.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message['content'].strip()

# Example conclusion
conclusion = "The cat is wet."
result = backward_reasoning(conclusion)
print(result)
