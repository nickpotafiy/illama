import asyncio
import os
import sys
import threading
import time
import unittest
from openai import OpenAI, AsyncOpenAI
import torch

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

        self.embed_earth = torch.tensor(
            [0.02, 0.33, -0.00, -0.31, -0.21, -0.13, -0.02, 0.28, -0.23, -0.17]
        )
        self.embed_macbook = torch.tensor(
            [-0.06, -0.08, 0.87, -0.53, 1.46, 0.17, 0.08, 0.37, 0.02, 0.38]
        )

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

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)

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

    def embed_close(self, tensor1, tensor2) -> bool:
        if not isinstance(tensor1, torch.Tensor):
            tensor1 = torch.tensor(tensor1)
        if not isinstance(tensor2, torch.Tensor):
            tensor2 = torch.tensor(tensor2)
        return torch.all(torch.isclose(tensor1, tensor2, atol=1e-2))

    def test_embedding_request(self):
        texts = ["europe", "apple macbook", "earth", "the holy bible"]
        for i, text in enumerate(texts):
            embeddings = self.client.embeddings.create(
                model=self.test_model, input=text
            )
            embeddings = embeddings.data[0].embedding
            self.assertEqual(len(embeddings), 4096)
            if i == 2:
                assert self.embed_close(
                    embeddings[0:10], self.embed_earth
                ), "embeddings do not match"
            if i == 1:
                assert self.embed_close(
                    embeddings[0:10], self.embed_macbook
                ), "embeddings do not match"

    def test_embedding_request_async(self):
        async def async_test():
            texts = ["the holy bible", "apple macbook", "earth", "europe"]
            tasks = []
            for text in texts:
                task = asyncio.create_task(
                    self.client_async.embeddings.create(
                        model=self.test_model, input=text
                    )
                )
                tasks.append(task)

            results = []
            for task in tasks:
                result = await task
                results.append(result)

            for i, result in enumerate(results):
                embedding = result.data[0].embedding
                self.assertEqual(len(embedding), 4096)
                if i == 1:
                    assert self.embed_close(
                        embedding[0:10], self.embed_macbook
                    ), "embeddings do not match"
                elif i == 2:
                    assert self.embed_close(
                        embedding[0:10], self.embed_earth
                    ), "embeddings do not match"

        self.run_async(async_test())


if __name__ == "__main__":
    unittest.main()
