import openai
from key import get_personal_key
# Method Explanation:
# Deductive reasoning involves drawing specific conclusions from a general set of premises.
# In NLP, this method allows a model to infer conclusions that are logically consistent
# with the given premises. For example, if the premise states "All mammals can breathe"
# and "A cat is a mammal," the model should deduce that "A cat can breathe."
# This type of reasoning is foundational for logical inference tasks, often tested using
# datasets like RuleTaker or ProofWriter.


# Set up your OpenAI API key
openai.api_key = get_personal_key()


def deductive_reasoning(premises: str) -> str:
    prompt = f"Given the following premises:\n\n{premises}\n\nWhat logical conclusion can you draw?"
    response = openai.ChatCompletion.create(
        model='gpt-4.0',
        messages=[
            {'role': 'system', 'content': 'You are a logical reasoner.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message['content'].strip()

# Example premises
premises = "1. All mammals can breathe.\n2. A cat is a mammal."
result = deductive_reasoning(premises)
print(result)
