from transformers import AutoModelForCausalLM, AutoConfig
from PIL import Image
from VLMConfig import VLMConfig, VLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = ''
AutoConfig.register("Qwenov3", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.bfloat16, trust_remote_code=True).to(device)
model.eval()
processor = model.processor
tokenizer = model.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

q_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": '<image>\n用中文描述图片内容。'}
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False).replace('<image>', '<|vision_start|>' + '<|image_pad|>' * model.config.image_pad_num + '<|vision_end|>')


text_inputs = tokenizer(q_text, return_tensors='pt')
input_ids = text_inputs['input_ids'].to(device)
attention_mask = text_inputs['attention_mask'].to(device)

image = Image.open('./A (3).jpg')
pixel_values = processor(images=image, return_tensors="pt")['pixel_values'].to(device)


max_new_tokens = 512
min_new_tokens = 10
num_beams = 4
repetition_penalty = 1.1

text_embeds = model.llm_model.get_input_embeddings()(input_ids)
image_embeds = model.vision_model(pixel_values).last_hidden_state
patch_embeds = image_embeds[:, 5:, :]
b, num_patches, hidden_dim = patch_embeds.shape
patch_embeds = patch_embeds.view(b, num_patches // 4, hidden_dim * 4)
image_features = model.adapter(patch_embeds)
text_embeds = text_embeds.to(image_features.dtype)
inputs_embeds = model.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)

with torch.inference_mode():
    output_ids = model.llm_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        # num_beams=num_beams,  # Using beam search may get stuck in a loop#
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
