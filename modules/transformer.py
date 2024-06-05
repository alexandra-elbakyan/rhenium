from torch import cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from quanto import freeze, quantize, qint8
from sentence_transformers import SentenceTransformer
from threading import Thread

def sentence(path):
    return SentenceTransformer(path, trust_remote_code = True, device = ["cpu", "cuda"][cuda.is_available()])

def generator(path, quanti = None):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code = True)
    if quanti:
        quantize(model, weights = qint8, activations = qint8)
        freeze(model)
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
