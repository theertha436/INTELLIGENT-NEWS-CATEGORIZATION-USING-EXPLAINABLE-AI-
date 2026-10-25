
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading translation model. The first run downloads model files...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang="tel_Telu"
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

text = (
    "రైతులకు సహాయం చేయడానికి ప్రభుత్వం "
    "కొత్త వ్యవసాయ పథకాన్ని ప్రారంభించింది."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_new_tokens=150
    )

print("Telugu:", text)
print("English:", tokenizer.decode(output[0], skip_special_tokens=True))
