import nltk
import regex as re
import torch

from nltk.corpus import stopwords
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM
import time
import regex as re
import torch

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"
use_triton = False
import numpy as np
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
import gensim
embeddings = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

stop_words = set(stopwords.words('english'))
ques2cntxt = torch.load('ques2cntxt.pt')
ques2wrds = torch.load('ques2wrds.pt')
# set up word2vec pretrained 300 dim

def rank_questions(questions, query):
    max_cos = 0
    max_ques = ""
    query_vec = np.zeros(300)
    for word in query.split():
        if word in embeddings:
            query_vec += embeddings[word]
        else:
            query_vec += embeddings["unk"]
    # divide by length of question
    query_vec /= len(query.split())

    for question in questions:
        # create empty vector of 300 dim
        ques_vec = np.zeros(300)
        for word in question.split():
            if word in embeddings:
                ques_vec += embeddings[word]
            else:
                ques_vec += embeddings["unk"]
        # divide by length of question
        ques_vec /= len(question.split())
        cos = np.dot(query_vec, ques_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(ques_vec))
        if cos > max_cos:
            max_cos = cos
            max_ques = question
        # delete all vectors saved on memory (ram)
    del ques_vec
    del query_vec
    del questions
    del cos
    return max_ques


        

def get_final_evidence(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    query_words = query.split()
    query_words = set(query_words)
    query_words = query_words - stop_words
    print("key words: ", query_words)
    max_score = 0
    for item in ques2wrds.keys():
        ques_set = ques2wrds[item]
        common_words = query_words.intersection(ques_set)
        length = len(common_words)
        if length > max_score:
            max_score = length
    questions = []
    for ques in ques2wrds.keys():
        ques_set = ques2wrds[ques]
        common_words = query_words.intersection(ques_set)
        length = len(common_words)
        if max_score >=2:
            if length == max_score:
                questions.append(ques)
        else:
            if length == max_score:
                questions.append(ques)
    top_question = rank_questions(questions, query)
    final_context = ques2cntxt[top_question]
    # contexts_set = set()
    # for ques in questions:
    #     contexts_set.add(ques2cntxt[ques])
    # print(len(contexts_set), "final context set")
    # final_context  = ""
    # for context in contexts_set:
    #     final_context += context
    #     final_context += " "
        
    # words_list = final_context.split()
    # final_context = " "
    # for word in words_list:
    #     if word not in stop_words:
    #         final_context += word
    #         final_context += " "
    # print(len(final_context.split()), "final Length...")
    return final_context

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
print("Done Setting Up, starting chatbot...")
while True:
    # Create the model

    system_message = "You will answer only based on the context provided. If the topic of the query is different from the context, please answer Out Of Context."
    prompt = input("You: ")
    if prompt == "":
        continue
    context = get_final_evidence(prompt)
    if context == "":
        continue
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