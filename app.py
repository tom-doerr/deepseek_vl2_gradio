import torch
import gradio as gr
import argparse
import re
from transformers import AutoModelForCausalLM
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image, ImageDraw, ImageFont
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

def draw_bounding_boxes(image, text):
    """
    Parse detection tags from text and draw bounding boxes on the image
    Format: <|det|>object_name<box>x1,y1,x2,y2</box><|/det|>
    """
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Regular expression to find detection tags
    det_pattern = r'<\|det\|>(.*?)<box>(.*?)</box><\|/det\|>'
    detections = re.findall(det_pattern, text)
    
    # Colors for different objects (cycling through)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for idx, (obj_name, box_coords) in enumerate(detections):
        # Parse coordinates
        try:
            x1, y1, x2, y2 = map(float, box_coords.split(','))
            # Convert relative coordinates to absolute if needed
            width, height = image.size
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            
            # Draw rectangle
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            draw.text((x1, y1-15), obj_name, fill=color)
            
        except ValueError as e:
            print(f"Error parsing coordinates for {obj_name}: {e}")
    
    return draw_image

def process_image_and_prompt(images, prompt, model_size):
    if not images:
        return "Please upload at least one image.", []
    
    output_images = []
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

        # Convert uploaded images to PIL format and store them
        pil_images = []
        for img in images:
            if img is not None and hasattr(img, 'name'):
                try:
                    pil_img = Image.open(img.name)
                    pil_images.append(pil_img)
                except Exception as e:
                    return f"Error loading image: {str(e)}", []
        
        if not pil_images:
            return "No valid images were uploaded.", []
            
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
        # Process each image and draw bounding boxes based on the answer
        output_images = []
        for img in images:
            if img is not None and hasattr(img, 'name'):
                try:
                    pil_img = Image.open(img.name)
                    annotated_img = draw_bounding_boxes(pil_img, answer)
                    output_images.append(annotated_img)
                except Exception as img_error:
                    print(f"Error processing image: {str(img_error)}")
        
        if not output_images:
            return answer, []
        return answer, output_images
    except Exception as e:
        return f"Error: {str(e)}", []

# Parse arguments first
parser = argparse.ArgumentParser(description='DeepseekVL Demo')
parser.add_argument('--model', choices=list(MODEL_PATHS.keys()), 
                   help='Fix the model size (disables model selection in UI)')
args = parser.parse_args()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# DeepSeek-VL2 Visual Language Model Demo")
    
    with gr.Row():
        with gr.Column():
            model_choice = gr.Radio(
                choices=[args.model] if args.model else list(MODEL_PATHS.keys()),
                value=args.model if args.model else "small",
                label="Model Size",
                info="Choose the model size (larger = better but slower)",
                interactive=not bool(args.model)
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
            image_output = gr.Gallery(label="Processed Images")
    
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, text_input, model_choice],
        outputs=[output, image_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
