from abc import ABC

from transformers import GPT2Model, GPT2Config
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_, **__):
        return None


class AutoModel:
    @classmethod
    def from_pretrained(cls, *_, **__):
        class MockModel(GPT2Model, ABC):
            @classmethod
            def stream_chat(cls, _, query, history) -> list:
                from time import sleep
                current_response = ''
                for i in range(3):
                    current_response += str(i)
                    yield current_response, history + [[query, current_response]]
                    sleep(1)

            def cuda(self, *args, **kwargs):
                return self

        return MockModel(GPT2Config())
