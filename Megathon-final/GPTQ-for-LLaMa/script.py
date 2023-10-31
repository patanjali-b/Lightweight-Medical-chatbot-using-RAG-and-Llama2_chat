from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM
import time
import regex as re
import torch

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"
use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def clear_cuda_memory():
    torch.cuda.empty_cache()
    for i in range(16):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def create_model():
    return AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None
    )
while True:
    # Create the model
    model = create_model()

    # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
    logging.set_verbosity(logging.CRITICAL)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=250,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    system_message = "You will answer only based on the context provided. If the topic of the query is different from the context, please answer Out Of Context."
    prompt = input("You: ")
    if prompt == "":
        continue
    context = input("Context: ")
    if context == "":
        continue
    system_message = system_message + "\n" + "Context: " + context + system_message
    prompt_template = f'''[INST] <<SYS>>
        {system_message}
        <</SYS>>
        {prompt} [/INST]'''
    start = time.time()
    response = pipe(prompt_template)
    end = time.time()
    print("Time taken: ", end - start)
    response = response[0]["generated_text"].split("[/INST]")[1]

    # Remove tabs and continuous multiple spaces from the response
    response = re.sub(r'[\t ]+', ' ', response)

    if response == None or response == "" or response == " ":
        response = "Out of Context!"

    print("==========================================================================================================")
    print("Bot: ", response)
    print("==========================================================================================================")

    # Clear CUDA memory in each iteration
    clear_cuda_memory()