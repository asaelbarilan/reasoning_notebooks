import openai


# Method Explanation:
# Multi-agent collaboration involves multiple AI agents working together to solve a problem.
# Each agent may bring a different perspective or specialize in a different aspect of the problem,
# and through collaboration, they can arrive at a solution that might be more complex or accurate
# than what a single agent could achieve. This method is particularly useful in scenarios
# that require diverse expertise or perspectives, and it is often implemented in systems
# that require negotiation, conflict resolution, or complex decision-making.


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Phi-3.5-mini-instruct model
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Method Explanation:
# Multi-agent collaboration involves multiple AI agents working together to solve a problem.

def query_agent(prompt: str, agent_id: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def multi_agent_collaboration(task: str, num_agents: int = 3) -> str:
    agents_responses = []
    for i in range(num_agents):
        response = query_agent(f'Please solve the following task: {task}', agent_id=i + 1)
        agents_responses.append(response)

    final_answer = 'Agent Collaboration Results:\n\n'
    for i, response in enumerate(agents_responses):
        final_answer += f'Agent {i + 1} Response:\n{response}\n\n'

    synthesis_prompt = "Based on all agents' responses, synthesize the final answer."
    final_answer += 'Synthesized Final Answer:\n' + query_agent(synthesis_prompt, agent_id='Synthesizer')
    return final_answer


# Example task
task = "How can we improve renewable energy adoption worldwide?"
result = multi_agent_collaboration(task)
print(result)
print('')
