import torch
import gradio as gr
from transformers import AutoModelForCausalLM
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import os

# Model initialization will be done in the processing function
MODEL_PATHS = {
    "tiny": "deepseek-ai/deepseek-vl2-tiny",
    "small": "deepseek-ai/deepseek-vl2-small",
    "base": "deepseek-ai/deepseek-vl2"
}

def load_model(model_size):
    model_path = MODEL_PATHS[model_size]
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_chat_processor, tokenizer, vl_gpt

def process_image_and_prompt(images, prompt, model_size):
    try:
        # Load model based on selected size
        vl_chat_processor, tokenizer, vl_gpt = load_model(model_size)
        
        # Prepare conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Convert uploaded images to PIL format
        pil_images = [Image.open(img.name) for img in images]
        
        # Prepare inputs
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        # Get image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# DeepseekVL Visual Language Model Demo")
    
    with gr.Row():
        with gr.Column():
            model_choice = gr.Radio(
                choices=list(MODEL_PATHS.keys()),
                value="small",
                label="Model Size",
                info="Choose the model size (larger = better but slower)"
            )
            image_input = gr.File(
                label="Upload Images",
                file_count="multiple",
                type="filepath"
            )
            text_input = gr.Textbox(
                label="Enter your prompt",
                placeholder="Describe what you want to know about the image(s)..."
            )
            submit_btn = gr.Button("Generate Response")
        
        with gr.Column():
            output = gr.Textbox(label="Model Response")
    
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, text_input, model_choice],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
