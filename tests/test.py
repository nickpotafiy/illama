import asyncio
import os
import sys
import threading
import time
import unittest
import numpy as np
from openai import OpenAI, AsyncOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from illama.illama import IllamaServer


class Tests(unittest.TestCase):

    test_host = "127.0.0.1"
    test_port = 5050
    test_model = "F:\Meta-Llama-3-8B-Instruct"

    @classmethod
    def setUpClass(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.client = OpenAI(api_key="no_api_key")
        self.client.base_url = f"http://{self.test_host}:{self.test_port}/v1"

        self.client_async = AsyncOpenAI(api_key="no_api_key")
        self.client_async.base_url = f"http://{self.test_host}:{self.test_port}/v1"

        self.server = IllamaServer(
            self.test_host, self.test_port, self.test_model, 2, verbose=False
        )
        self.server_thread = threading.Thread(target=self.server.serve, daemon=True)
        self.server_thread.start()
        while not self.server.running:
            time.sleep(1)

    @classmethod
    def tearDownClass(self):
        self.loop.close()
        if self.server:
            self.server.stop()

    def test_models_request(self):
        models = self.client.models.list()
        models = models.data
        model = models[0]
        id = model.id
        object = model.object
        assert "Meta-Llama-3-8B-Instruct" == id
        assert object == "model"

    def test_completions_request_non_streaming(self):
        completion = self.client.chat.completions.create(
            model=self.test_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Follow the user's instructions carefully.",
                },
                {
                    "role": "user",
                    "content": "Please list days of the week starting with monday, all the way to sunday.",
                },
            ],
            max_tokens=40,
            stream=False,
            temperature=0.10,
        )
        content = completion.choices[0].message.content
        assert (
            "monday" in content.lower()
        ), f"Could not find expected string 'monday' in '{content}'"
        assert (
            "tuesday" in content.lower()
        ), f"Could not find expected string 'tuesday' in '{content}'"

    def test_completions_request_streaming(self):
        completion = self.client.chat.completions.create(
            model=self.test_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Follow the user's instructions carefully.",
                },
                {
                    "role": "user",
                    "content": "Please list days of the week starting with monday, all the way to sunday.",
                },
            ],
            max_tokens=40,
            stream=True,
            temperature=0.10,
        )
        content = ""
        for message in completion:
            token = message.choices[0].delta.content
            if token is not None:
                content += token
            print(token, end="", flush=True)
        assert (
            "monday" in content.lower()
        ), f"Could not find expected string 'monday' in '{content}'"
        assert (
            "tuesday" in content.lower()
        ), f"Could not find expected string 'tuesday' in '{content}'"


if __name__ == "__main__":
    unittest.main()
