from transformers import AutoModelForCausalLM, AutoConfig
from PIL import Image
from VLMConfig import VLMConfig, VLM
import torch
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = ''
AutoConfig.register("Qwenov3", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.bfloat16).to(device)
model.eval()
processor = model.processor
tokenizer = model.tokenizer

q_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": '<image>\n描述图片内容。'}
    ],
    tokenize=False,
    add_generation_prompt=True, enable_thinking=False).replace('<image>', '<|vision_start|>' + '<|image_pad|>' * model.config.image_pad_num + '<|vision_end|>')

input_ids = tokenizer(q_text, return_tensors='pt')['input_ids'].to(device)

image = Image.open('./A (3).jpg')
pixel_values = processor(images=image, return_tensors="pt")['pixel_values'].to(device)


max_new_tokens = 512
temperature = 0.8
eos = tokenizer.eos_token_id
top_k = 20
top_p = 0.95
s = input_ids.shape[1]
while input_ids.shape[1] < s + max_new_tokens - 1:
    inference_res = model(input_ids, None, pixel_values)
    logits = inference_res.logits
    logits = logits[:, -1, :]

    for token in set(input_ids.tolist()[0]):
        logits[:, token] /= 1.0

    if temperature == 0.0:
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative_probs > top_p
            cutoff_mask[..., 0] = False
            sorted_probs[cutoff_mask] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(probs, num_samples=1, generator=None)

    if idx_next == eos:
        break

    input_ids = torch.cat((input_ids, idx_next), dim=1)
print(tokenizer.decode(input_ids[:, s:][0]))
