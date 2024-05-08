from torch import cuda
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def sentence(path):
    return SentenceTransformer(path, trust_remote_code = True, device = ["cpu", "cuda"][cuda.is_available()])

def generator(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    if cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def answer(tokenizer, model, question):
    inputs = tokenizer(question, return_tensors = "pt", return_attention_mask = False)
    if cuda.is_available():
        inputs = inputs.to("cuda")
    outputs = model.generate(**inputs, max_length = 256, pad_token_id = 100)
    text = tokenizer.batch_decode(outputs)[0]
    return text
