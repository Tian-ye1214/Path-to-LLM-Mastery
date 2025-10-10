from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from modelscope import AutoConfig, AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class Qwenov3Config(PretrainedConfig):
    model_type = "Qwenov3"

    def __init__(self, llm_model_path='Qwen/Qwen3-0.6B',
                 vision_model_path='facebook/dinov3-vitl16-pretrain-lvd1689m',
                 freeze_vision_model=False,
                 freeze_llm_model=False,
                 image_pad_num=49,
                 training_scratch=False,
                 num_hidden_layers=None,
                 hidden_size=None,
                 num_attention_heads=None,
                 vocab_size=None,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm_model = freeze_llm_model
        self.image_pad_num = image_pad_num
        self.freeze_vision_model = freeze_vision_model
        self.training_scratch = training_scratch
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        
        super().__init__(**kwargs)


class Qwenov3(GenerationMixin, PreTrainedModel):
    config_class = Qwenov3Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.training_scratch:
            self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path, low_cpu_mem_usage=True,
                                                          dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, low_cpu_mem_usage=True,
                                                                  dtype=torch.bfloat16,
                                                                  attn_implementation="flash_attention_2")
        else:
            vision_config = AutoConfig.from_pretrained(self.config.vision_model_path)
            self.vision_model = AutoModel.from_config(vision_config, attn_implementation="sdpa", dtype=torch.bfloat16)
            llm_config = AutoConfig.from_pretrained(self.config.llm_model_path)
            self.llm_model = AutoModelForCausalLM.from_config(llm_config, attn_implementation="sdpa", dtype=torch.bfloat16)

        if self.config.num_hidden_layers is None:
            self.config.num_hidden_layers = self.llm_model.config.num_hidden_layers
        if self.config.hidden_size is None:
            self.config.hidden_size = self.llm_model.config.hidden_size
        if self.config.num_attention_heads is None:
            self.config.num_attention_heads = self.llm_model.config.num_attention_heads
        if self.config.vocab_size is None:
            self.config.vocab_size = self.llm_model.config.vocab_size

        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if '<|image_pad|>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(['<|image_pad|>'])
            self.llm_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
        if '<|vision_start|>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(['<|vision_start|>'])
            self.llm_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
        if '<|vision_end|>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(['<|vision_end|>'])
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

    def forward(self, input_ids=None, labels=None, pixel_values=None, attention_mask=None, 
                inputs_embeds=None, past_key_values=None, use_cache=None, **kwargs):

        if inputs_embeds is None:
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

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True
        )
        
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        
        return CausalLMOutputWithPast(
            loss=loss, 
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    @torch.inference_mode()
    def generate(self, input_ids=None, pixel_values=None, attention_mask=None, 
                 max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20,
                 do_sample=True, num_beams=1, use_cache=True, **kwargs):
        if pixel_values is not None:
            text_embeds = self.llm_model.get_input_embeddings()(input_ids)
            image_embeds = self.vision_model(pixel_values).last_hidden_state
            patch_embeds = image_embeds[:, 5:, :]
            b, num_patches, hidden_dim = patch_embeds.shape
            patch_embeds = patch_embeds.view(b, num_patches // 4, hidden_dim * 4)
            image_features = self.adapter(patch_embeds)
            text_embeds = text_embeds.to(image_features.dtype)
            inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
            return self.llm_model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        else:
            return self.llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_beams=num_beams,
                use_cache=use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

    def can_generate(self):
        return True

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        if len(batch_indices) == 0:
            return inputs_embeds
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        return inputs_embeds
