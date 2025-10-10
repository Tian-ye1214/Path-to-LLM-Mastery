import streamlit as st
import torch
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoConfig
import gc
from threading import Thread
from Qwenov3Config import Qwenov3Config, Qwenov3
from PIL import Image

MODEL_MAPPING = {
    'QwenoV3-Pretrain': '',
    'QwenoV3-SFT': 'TianYeZ1214/Qwenov3',
}


def unload_model():
    if 'model' in st.session_state:
        del st.session_state.model
    if 'tokenizer' in st.session_state:
        del st.session_state.tokenizer
    if 'processor' in st.session_state:
        del st.session_state.processor
    if 'streamer' in st.session_state:
        del st.session_state.streamer
    torch.cuda.empty_cache()
    gc.collect()


def call_model(info_placeholder, messages, generated_text, message_placeholder, image=None):
    info_placeholder.markdown(f'已选择{st.session_state.model_display}执行任务')
    if image is not None:
        image = Image.open(image).convert('RGB')
        if '<image>' not in messages[1]['content']:
            messages[1]['content'] = '<image>\n' + messages[1]['content']

    query_text = st.session_state.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    if '<image>' in query_text:
        query_text = query_text.replace('<image>', '<|vision_start|>' + '<|image_pad|>' *
                                        st.session_state.model.config.image_pad_num + '<|vision_end|>')
    text_inputs = st.session_state.tokenizer(query_text, return_tensors="pt")
    input_ids = text_inputs['input_ids'].to(st.session_state.model.device)
    attention_mask = text_inputs['attention_mask'].to(st.session_state.model.device)
    text_embeds = st.session_state.model.llm_model.get_input_embeddings()(input_ids)

    if image is not None:
        pixel_values = st.session_state.processor(images=image, return_tensors="pt")['pixel_values'].to(
            st.session_state.model.device)
        image_embeds = st.session_state.model.vision_model(pixel_values).last_hidden_state
        patch_embeds = image_embeds[:, 5:, :]
        b, num_patches, hidden_dim = patch_embeds.shape
        patch_embeds = patch_embeds.view(b, num_patches // 4, hidden_dim * 4)
        image_features = st.session_state.model.adapter(patch_embeds)
        text_embeds = text_embeds.to(image_features.dtype)
        inputs_embeds = st.session_state.model.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
    else:
        inputs_embeds = text_embeds

    generate_params = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=st.session_state.max_new_tokens,
        min_new_tokens=st.session_state.min_new_tokens,
        do_sample=True,
        temperature=st.session_state.temperature,
        top_k=st.session_state.top_k,
        top_p=st.session_state.top_p,
        min_p=0.0,
        repetition_penalty=st.session_state.repetition_penalty,
        streamer=st.session_state.streamer,
        eos_token_id=st.session_state.tokenizer.eos_token_id
    )
    thread = Thread(target=st.session_state.model.llm_model.generate, kwargs=generate_params)
    thread.start()

    for new_text in st.session_state.streamer:
        generated_text += new_text
        message_placeholder.markdown(generated_text)

    return generated_text


def ini_message():
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are QwenoV3, a helpful assistant created by 天烨."},
        ]
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None


def parameter_settings():
    with st.sidebar:
        previous_model = st.session_state.get('model_display', None)
        st.session_state.model_display = st.selectbox("选择模型", list(MODEL_MAPPING.keys()),
                                                      index=len(MODEL_MAPPING.keys()) - 1, help="选择模型")
        st.session_state.model_path = MODEL_MAPPING[st.session_state.model_display]
        with st.expander("对话参数", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                                                         help="控制模型回答的多样性，值越高表示回复多样性越高")
                st.session_state.min_new_tokens = st.number_input("Min Tokens",
                                                                  min_value=0,
                                                                  max_value=512,
                                                                  value=10,
                                                                  help="生成文本的最小长度")
                st.session_state.max_new_tokens = st.number_input("Max Tokens",
                                                                  min_value=1,
                                                                  max_value=4096,
                                                                  value=512,
                                                                  help="生成文本的最大长度")
            with col2:
                st.session_state.top_p = st.slider("Top P", 0.0, 1.0, 0.8, 0.1,
                                                   help="控制词汇选择的多样性,值越高表示潜在生成词汇越多样")
                st.session_state.top_k = st.slider("Top K", 0, 80, 20, 1,
                                                   help="控制词汇选择的多样性,值越高表示潜在生成词汇越多样")
                st.session_state.repetition_penalty = st.slider("Repetition Penalty", 0.0, 2.0, 1.05, 0.1,
                                                                help="控制回复主题的多样性性，值越高重复性越低")

        with st.expander("图片上传", expanded=False):
            st.session_state.uploaded_image = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png"]
            )
            if st.session_state.uploaded_image:
                image = Image.open(st.session_state.uploaded_image)
                width, height = image.size
                if width > 256 or height > 256:
                    scale = 256 / max(height, width)
                    new_h, new_w = int(height * scale), int(width * scale)
                    image = image.resize((new_w, new_h), Image.BILINEAR)
                st.image(image, caption="图片预览")

        if st.button("开启新对话", help="开启新对话将清空当前对话记录"):
            st.session_state.uploaded_image = None
            st.session_state.messages = [
                {"role": "system", "content": "You are QwenoV3, a helpful assistant created by 天烨."},
            ]
            st.success("已成功开启新的对话")
            st.rerun()

        if previous_model != st.session_state.model_display or 'tokenizer' not in st.session_state or 'model' not in st.session_state or 'processor' not in st.session_state:
            unload_model()
            try:
                with st.spinner('加载模型中...'):
                    AutoConfig.register("Qwenov3", Qwenov3Config)
                    AutoModelForCausalLM.register(Qwenov3Config, Qwenov3)
                    st.session_state.model = AutoModelForCausalLM.from_pretrained(
                        st.session_state.model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    st.session_state.tokenizer = st.session_state.model.tokenizer
                    st.session_state.processor = st.session_state.model.processor
                    st.session_state.streamer = TextIteratorStreamer(st.session_state.tokenizer,
                                                                     skip_prompt=True, skip_special_tokens=True)
            except Exception as e:
                st.error('模型加载出错：', e)
                return


def main():
    st.markdown("""
    <h1 style='text-align: center;'>
        QwenoV3 - Marrying DinoV3 With Qwen3 🫡
    </h1>
    <div style='text-align: center; margin-bottom: 20px;'>
    </div>
    """, unsafe_allow_html=True)
    ini_message()
    parameter_settings()

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("在这里输入您的问题：", key="chat_input"):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            info_placeholder = st.empty()
            message_placeholder = st.empty()
            generated_text = ""
            try:
                with torch.inference_mode():
                    generated_text = call_model(info_placeholder, st.session_state.messages, generated_text,
                                                message_placeholder, st.session_state.uploaded_image)
                st.session_state.messages.append({"role": "assistant", "content": generated_text})
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")


if __name__ == '__main__':
    main()
