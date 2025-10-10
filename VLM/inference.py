from transformers import AutoModelForCausalLM, AutoConfig
from PIL import Image
from Qwenov3Config import Qwenov3Config, Qwenov3
import torch
from transformers.image_utils import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'TianYeZ1214/Qwenov3'
AutoConfig.register("Qwenov3", Qwenov3Config)
AutoModelForCausalLM.register(Qwenov3Config, Qwenov3)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)
model.eval()

processor = model.processor
tokenizer = model.tokenizer

messages = [
    {"role": "system", "content": 'You are a helpful assistant.'},
    {"role": "user", "content": '<image>\n用中文描述图片内容。'},
]

if '<image>' not in messages[1]['content']:
    messages[1]['content'] = '<image>\n' + messages[1]['content']


q_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
).replace(
    '<image>',
    '<|vision_start|>' + '<|image_pad|>' * model.config.image_pad_num + '<|vision_end|>'
)

text_inputs = tokenizer(q_text, return_tensors='pt')
input_ids = text_inputs['input_ids'].to(device)
attention_mask = text_inputs['attention_mask'].to(device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
pixel_values = processor(images=image, return_tensors="pt")['pixel_values'].to(device)

output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    max_new_tokens=512,
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    do_sample=True,
    repetition_penalty=1.00,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
