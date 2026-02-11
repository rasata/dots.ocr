import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

def inference(image_path, prompt, model, processor):
    # image_path = "demo/demo_image1.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=24000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)



def get_device_config():
    """Detect the best available device and return (device, dtype, attn_implementation)."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16, "flash_attention_2"
    elif torch.backends.mps.is_available():
        return "mps", torch.float16, "sdpa"
    else:
        return "cpu", torch.float32, "sdpa"


if __name__ == "__main__":
    device, dtype, attn_impl = get_device_config()
    print(f"Using device: {device}, dtype: {dtype}, attention: {attn_impl}")

    model_path = "./weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device != "cuda":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True)

    image_path = "demo/demo_image1.jpg"
    for prompt_mode, prompt in dict_promptmode_to_prompt.items():
        print(f"prompt: {prompt}")
        inference(image_path, prompt, model, processor)
    