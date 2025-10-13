from transformers import AutoModelForCausalLM, AutoConfig
from transformers.image_utils import load_image
from Qwenov3Config import Qwenov3Config, Qwenov3, Qwenov3Processor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'TianYeZ1214/Qwenov3'
AutoConfig.register("Qwenov3", Qwenov3Config)
AutoModelForCausalLM.register(Qwenov3Config, Qwenov3)

model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.bfloat16,
                                             trust_remote_code=True, attn_implementation="flash_attention_2").to(device)
processor = Qwenov3Processor(image_processor=model.processor, tokenizer=model.tokenizer)
model.eval()

messages = [
    {"role": "system", "content": 'You are a helpful assistant.'},
    {"role": "user", "content": "描述图片内容"},
]

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

q_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = processor(
    text=[q_text],
    images=image,
    padding=True,
    return_tensors="pt",
).to(device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    do_sample=True,
    repetition_penalty=1.1,
)

output_ids = output_ids[0].tolist()

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

content = processor.decode(output_ids[index:], skip_special_tokens=True)
print("content:", content)
