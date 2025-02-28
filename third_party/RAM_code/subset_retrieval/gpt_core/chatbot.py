import os
from openai import OpenAI
from subset_retrieval.gpt_core.system_message import language_system_part1, language_system_part2

class Chatbot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def run_one_round_conversation(self, message, model_name="gpt-3.5-turbo-1106"):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=model_name,
            temperature=0,
        )

        return response.choices[0].message.content
    
    def task_retrieval(self, task_list, current_task):
        # query ChatGPT to decide which task to retrieve
        message = language_system_part1 + "\n".join(task_list) + language_system_part2
        message = message + "\n" + current_task

        response_message = self.run_one_round_conversation(
            message
        )

        for task in task_list:
            if task in response_message:
                return task
            
        return None