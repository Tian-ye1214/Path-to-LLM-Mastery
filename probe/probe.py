from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Option
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_name = ""

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    # quantization_config=quantization_config,
    attn_implementation="sdpa",
)

prompt = "介绍一下你自己"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    output_hidden_states=True,
    return_dict_in_generate=True,
)
generated_ids = outputs.sequences
hidden_states = outputs.hidden_states

m = len(hidden_states[0])
n = len(hidden_states)

all_layer_states = [
    torch.cat([hidden_states[i][j][:, -1, :] for i in range(n)], dim=0)
    for j in range(m)
]

output_tensor = torch.stack(all_layer_states, dim=0)
print(output_tensor.shape)


print("\nModel's state_dict keys:")
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}")



