import os
from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)


class OpenaiClient:
    def __init__(self, api_key=None, organization=None, project=None, base_url=None, model=None, max_tokens=8192):
        api_key  = api_key  or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, organization=organization, project=project, base_url=base_url)

    def get_chatgpt_response(self, prompt, content):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            top_p=1
        )

        return response.choices[0].message.content


    def get_chatgpt_response_with_file(self, prompt, filename):
        pdf_assistant = self.client.beta.assistants.create(
            model=self.model,
            description="An assistant to extract the contents of PDF files.",
            tools=[{"type": "file_search"}],
            name="PDF assistant",
        )

        # Create thread
        thread = self.client.beta.threads.create()
        file = self.client.files.create(file=open(filename, "rb"), purpose="assistants")

        # Create assistant
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[
                Attachment(
                    file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
                )
            ],
            content=prompt,
        )

        # Run thread
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
        )

        if run.status != "completed":
            raise Exception("Run failed:", run.status)

        messages_cursor = self.client.beta.threads.messages.list(thread_id=thread.id)
        messages = [message for message in messages_cursor]

        # Output text
        return messages[0].content[0].text.value
