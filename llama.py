import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, QuantoConfig

quantization_config = QuantoConfig(weights="int8")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="universeTBD/astrollama"
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="universeTBD/astrollama",
    device_map="auto",
    quantization_config=quantization_config
)

generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Taken from https://arxiv.org/abs/2308.12823
prompt = "In this letter, we report the discovery of the highest redshift, " \
    "heavily obscured, radio-loud QSO candidate selected using JWST NIRCam/MIRI, " \
    "mid-IR, sub-mm, and radio imaging in the COSMOS-Web field. "

# For reproducibility
torch.manual_seed(42)

generated_text = generator(
    prompt,
    do_sample=True,
    max_length=512
)

print(generated_text)

