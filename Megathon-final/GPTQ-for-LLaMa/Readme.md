### Directory Structure
```
.
├── convert_llama_weights_to_hf.py
├── final_rag.py
├── GoogleNews-vectors-negative300.bin
├── gptq.py
├── LICENSE.txt
├── llama-hf
│   └── llama-7b
│       └── tmp
├── llama_inference_offload.py
├── llama_inference.py
├── llama.py
├── neox.py
├── opt.py
├── quant
│   ├── custom_autotune.py
│   ├── fused_attn.py
│   ├── fused_mlp.py
│   ├── __init__.py
│   ├── quantizer.py
│   ├── quant_linear.py
│   └── triton_norm.py
├── ques2cntxt.pt
├── ques2wrds.pt
├── Readme.md
├── README.md
├── requirements.txt
├── requirements.yml
├── script.py
├── utils
│   ├── datautils.py
│   ├── export.py
│   ├── __init__.py
│   └── modelutils.py
└── word2vec.model

```
### set up the environment using requirements.yml not requirements.txt


### First set up hugging face cli, use your huggingface_token
```
pip install --upgrade huggingface_hub
huggingface-cli login --token $HUGGINGFACE_TOKEN

```
### Then get access to llama-2 model using this link: https://ai.meta.com/resources/models-and-libraries/llama-downloads/

### Download Word2Vec into the current working directory using: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300/download?datasetVersionNumber=2 or https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

### then run the following command to have the chatbot up and running:
```
python3 -W ignore final_rag.py
```

### TO run the application, run:
```
python3 app.py
```