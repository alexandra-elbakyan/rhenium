from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from threading import Thread

def sentence(path):
    return SentenceTransformer(path, trust_remote_code = True, device = ["cpu", "cuda"][cuda.is_available()])

def generator(path, quanti = None):
    if quanti:
       quanti = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_compute_dtype = bfloat16
                 )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code = True, quantization_config = quanti)
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
