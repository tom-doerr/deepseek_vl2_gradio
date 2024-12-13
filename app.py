import torch
import gradio as gr
from transformers import AutoModelForCausalLM
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import os

# Initialize model and processor
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def process_image_and_prompt(images, prompt):
    try:
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
            image_input = gr.File(
                label="Upload Images",
                file_count="multiple",
                type="file"
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
        inputs=[image_input, text_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
