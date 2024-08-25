import openai
from key import get_personal_key


# Method Explanation:
# Commonsense reasoning enables models to make inferences based on everyday knowledge that
# people typically take for granted. This type of reasoning is crucial for understanding
# and interacting with the world in a way that aligns with human expectations.
# Datasets like ART (Abductive Natural Language Inference) help assess how well models
# can perform commonsense reasoning by requiring them to infer plausible explanations
# for given situations.


# Set up your OpenAI API key
openai.api_key = get_personal_key()


def commonsense_reasoning(scenario: str) -> str:
    prompt = f"Given the following scenario:\n\n{scenario}\n\nWhat is the most plausible explanation or conclusion?"
    response = openai.ChatCompletion.create(
        model='gpt-4.0',
        messages=[
            {'role': 'system', 'content': 'You are a commonsense reasoner.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

# Example scenario
scenario = "John left the house with an umbrella even though it wasn't raining."
result = commonsense_reasoning(scenario)
print(result)
