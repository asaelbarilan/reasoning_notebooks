import openai
from key import get_personal_key
# Method Explanation:
# Defeasible reasoning is a form of reasoning where the conclusions drawn can be retracted
# if new information contradicts them. It allows models to dynamically adjust their conclusions
# based on additional evidence, making it particularly useful in real-world scenarios where
# not all information is initially available. This type of reasoning is explored in NLP using
# datasets that require models to revise their conclusions as new information is introduced.


# Set up your OpenAI API key
openai.api_key = get_personal_key()


def defeasible_reasoning(premises: str, new_information: str) -> str:
    prompt = f"Given the following premises:\n\n{premises}\n\nNow, consider this new information:\n\n{new_information}\n\nWhat conclusions can you draw now?"
    response = openai.ChatCompletion.create(
        model='gpt-4.0',
        messages=[
            {'role': 'system', 'content': 'You are a defeasible reasoner.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message['content'].strip()

# Example premises and new information
premises = "1. It is cloudy outside.\n2. When it is cloudy, it usually rains."
new_information = "However, the weather forecast says it will be sunny."
result = defeasible_reasoning(premises, new_information)
print(result)
