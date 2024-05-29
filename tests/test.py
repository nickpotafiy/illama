import os
import sys
import threading
import time
import unittest
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from illama.illama import IllamaServer


class Tests(unittest.TestCase):

    test_host = "127.0.0.1"
    test_port = 5050
    test_model = "F:\\Meta-Llama-3-8B-Instruct\\"

    @classmethod
    def setUpClass(self):
        self.client = OpenAI(api_key="no_api_key")
        self.client.base_url = f"http://{self.test_host}:{self.test_port}/v1"
        self.server = IllamaServer(
            self.test_host, self.test_port, self.test_model, 5, verbose=False
        )
        self.server_thread = threading.Thread(target=self.server.serve, daemon=True)
        self.server_thread.start()
        while not self.server.running:
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.stop()

    def test_models_request(self):
        models = self.client.models.list()
        models = models.data
        model = models[0]
        id = model.id
        object = model.object
        assert id == self.test_model
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

    def test_single_embedding_request(self):
        response = self.client.embeddings.create(
            model=self.test_model, input="monday"
        )
        self.assertEqual(len(response.data), 1)
        
        embedding = response.data[0].embedding
        self.assertEqual(len(embedding), 4096)

        response = self.client.embeddings.create(
            model=self.test_model, input="monday tuesday wednesday"
        )
        self.assertEqual(len(response.data), 1)
        
        embedding = response.data[0].embedding
        self.assertEqual(len(embedding), 4096)

    def test_multi_embedding_request(self):
        response = self.client.embeddings.create(
            model=self.test_model, input=["monday tuesday", "wednesday", "thrusday friday"]
        )
        self.assertEqual(len(response.data), 3)
        for _embedding in response.data:
            embedding = _embedding.embedding
            self.assertEqual(len(embedding), 4096)

if __name__ == "__main__":
    unittest.main()
