from typing import Optional, Union
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin, BatchFeature
from transformers.modeling_outputs import CausalLMOutputWithPast
from modelscope import AutoConfig, AutoProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers.image_utils import ImageInput
from liger_kernel.transformers import LigerCrossEntropyLoss, AutoLigerKernelForCausalLM, LigerLayerNorm
from transformers.processing_utils import Unpack, ProcessorMixin
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import TransformersKwargs


class Qwenov3Config(PretrainedConfig):
    model_type = "Qwenov3"

    def __init__(self,
                 llm_model_path='Qwen/Qwen3-0.6B',
                 vision_model_path='facebook/dinov3-vitl16-pretrain-lvd1689m',
                 freeze_vision_model=False,
                 freeze_llm_model=False,
                 image_pad_num=246,
                 training_scratch=False,
                 num_hidden_layers=28,
                 hidden_size=1024,
                 num_attention_heads=16,
                 vocab_size=151936,
                 vision_hidden_size=4096,
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
        self.vision_hidden_size = vision_hidden_size
        super().__init__(**kwargs)


class Qwenov3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, image_pad_num=246, **kwargs):
        self.image_token = "<|image_pad|>"
        self.image_pad_num = image_pad_num
        if chat_template is None and tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
            self,
            images: Optional[ImageInput] = None,
            text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
            return_tensors: str = "pt",
            **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        if images is not None:
            from Dataset import process_image_to_5_patches
            images = process_image_to_5_patches(images)
            pixel_values = self.image_processor(images=images, return_tensors="pt")['pixel_values']
            if len(pixel_values.shape) == 4:
                pixel_values = pixel_values.unsqueeze(0)
            image_inputs = {'pixel_values': pixel_values}
        if not isinstance(text, list):
            text = [text]
        processed_text = []
        if images is not None:
            for t in text:
                replacement = '<|vision_start|>' + '<|image_pad|>' * self.image_pad_num + '<|vision_end|>'
                if '<image>' not in t:
                    t = t.replace('<|im_start|>user', '<|im_start|>user\n<image>', 1)
                processed_text.append(t.replace('<image>', replacement))
        else:
            processed_text = text
        tokenizer_kwargs = {k: v for k, v in kwargs.items() if k not in ['images']}
        text_inputs = self.tokenizer(processed_text, return_tensors=return_tensors, **tokenizer_kwargs)
        return BatchFeature(data={**text_inputs, **image_inputs})


class Qwenov3(PreTrainedModel, GenerationMixin):
    config_class = Qwenov3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _tied_weights_keys = ["llm_model.lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.training_scratch:
            self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path,
                                                          low_cpu_mem_usage=True,
                                                          dtype=torch.bfloat16,
                                                          attn_implementation="sdpa")
            self.llm_model = AutoLigerKernelForCausalLM.from_pretrained(self.config.llm_model_path,
                                                                        low_cpu_mem_usage=True,
                                                                        dtype=torch.bfloat16,
                                                                        attn_implementation="sdpa", )
        else:
            vision_config = AutoConfig.from_pretrained(self.config.vision_model_path)
            self.vision_model = AutoModel.from_config(vision_config,
                                                      attn_implementation="sdpa",
                                                      dtype=torch.bfloat16)
            llm_config = AutoConfig.from_pretrained(self.config.llm_model_path)
            self.llm_model = AutoLigerKernelForCausalLM.from_config(llm_config,
                                                                    attn_implementation="sdpa",
                                                                    dtype=torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, use_fast=True)
        self.adapter = nn.Sequential(
            LigerLayerNorm(self.config.vision_hidden_size),
            nn.Linear(self.config.vision_hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

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

        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
        if self.config.freeze_llm_model:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model.eval()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            # pixel_values: [B, 5, 3, 224, 224]
            B, num_images, C, H, W = pixel_values.shape
            pixel_values_flat = pixel_values.view(B * num_images, C, H, W)
            image_embeds_flat = self.vision_model(pixel_values_flat).last_hidden_state
            image_embeds = image_embeds_flat.view(B, num_images, -1, image_embeds_flat.shape[-1])  # [B, 5, 201, 1024]

            batch_features = []
            for b in range(B):
                first_img_tokens = image_embeds[b, 0, 5:, :]
                other_img_tokens = []
                for i in range(1, num_images):
                    tokens = torch.cat([
                        image_embeds[b, i, 0:1, :],  # 第1个token [1, 1024]
                        image_embeds[b, i, 5:, :]    # 后196个token [196, 1024]
                    ], dim=0)  # [197, 1024]
                    other_img_tokens.append(tokens)
                
                # 拼接：[196, 1024] + 4*[197, 1024] = [984, 1024]
                sample_tokens = torch.cat([first_img_tokens] + other_img_tokens, dim=0)
                batch_features.append(sample_tokens)

            patch_embeds = torch.stack(batch_features, dim=0)
            _, total_tokens, hidden_dim = patch_embeds.shape
            patch_embeds = patch_embeds.view(B, total_tokens // 4, hidden_dim * 4)
            image_features = self.adapter(patch_embeds)
            text_embeds = text_embeds.to(image_features.dtype)
            inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        else:
            inputs_embeds = text_embeds

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
            loss_fct = LigerCrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device))
            loss = loss / 64.0

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(self, input_ids=None, pixel_values=None, attention_mask=None,
                 max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20,
                 do_sample=True, num_beams=1, use_cache=True, **kwargs):
        if pixel_values is not None:
            text_embeds = self.llm_model.get_input_embeddings()(input_ids)
            
            B, num_images, C, H, W = pixel_values.shape
            pixel_values_flat = pixel_values.view(B * num_images, C, H, W)
            image_embeds_flat = self.vision_model(pixel_values_flat, output_hidden_states=True).last_hidden_state
            image_embeds = image_embeds_flat.view(B, num_images, -1, image_embeds_flat.shape[-1])
 
            batch_features = []
            for b in range(B):
                first_img_tokens = image_embeds[b, 0, 5:, :]
                other_img_tokens = []
                for i in range(1, num_images):
                    tokens = torch.cat([
                        image_embeds[b, i, 0:1, :],
                        image_embeds[b, i, 5:, :]
                    ], dim=0)
                    other_img_tokens.append(tokens)
                
                sample_tokens = torch.cat([first_img_tokens] + other_img_tokens, dim=0)
                batch_features.append(sample_tokens)
 
            patch_embeds = torch.stack(batch_features, dim=0)
            _, total_tokens, hidden_dim = patch_embeds.shape
            patch_embeds = patch_embeds.view(B, total_tokens // 4, hidden_dim * 4)
            image_features = self.adapter(patch_embeds)
            text_embeds = text_embeds.to(image_features.dtype)
            inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        else:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        return self.llm_model.generate(
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

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)
        return inputs_embeds
