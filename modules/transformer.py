from torch import cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, QuantoConfig
from sentence_transformers import SentenceTransformer
from threading import Thread

def sentence(path):
    return SentenceTransformer(path, trust_remote_code = True, device = ["cpu", "cuda"][cuda.is_available()])

def generator(path, quantize = None):
    if quantize:
        quantize = QuantoConfig(**quantize)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config = quantize, trust_remote_code = True)
    if cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def answer(tokenizer, model, question):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt = True)
    inputs = tokenizer(question, return_tensors = "pt", return_attention_mask = False)
    if cuda.is_available():
        inputs = inputs.to("cuda")
    generation_kwargs = dict(inputs, streamer = streamer, max_new_tokens = 2048)
    thread = Thread(target = model.generate, kwargs = generation_kwargs)
    thread.start()
    return streamer
