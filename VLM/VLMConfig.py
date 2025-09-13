from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"

    def __init__(self, llm_model_path='/root/autodl-tmp/ModelCheckpoint/Qwen3',
                 vision_model_path='/root/autodl-tmp/ModelCheckpoint/Dinov3',
                 freeze_vision_model=False,
                 freeze_llm_model=False,
                 image_pad_num=49,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm_model = freeze_llm_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path, low_cpu_mem_usage=True,
                                                      dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, low_cpu_mem_usage=True,
                                                              dtype=torch.bfloat16,
                                                              attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if '<|image_pad|>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(['<|image_pad|>'])
            self.llm_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)

        self.adapter = nn.Sequential(
            nn.RMSNorm(4096, dtype=torch.bfloat16),
            nn.Linear(4096, self.llm_model.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size, dtype=torch.bfloat16)
        )

        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if self.config.freeze_llm_model:
            for param in self.llm_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values=None, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.vision_model(pixel_values).last_hidden_state
            patch_embeds = image_embeds[:, 5:, :]  # [batch, 196, 1024]
            b, num_patches, hidden_dim = patch_embeds.shape
            patch_embeds = patch_embeds.view(b, num_patches // 4, hidden_dim * 4)  # [batch, 49, 4096]
            image_features = self.adapter(patch_embeds)
            text_embeds = text_embeds.to(image_features.dtype)
            inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        else:
            inputs_embeds = text_embeds

        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):

        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])

        if len(batch_indices) == 0:
            return inputs_embeds

        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)

        return inputs_embeds
