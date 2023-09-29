import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

model_name = "NousResearch/Nous-Hermes-llama-2-7b"
adapters_name = "rebootai/ragama-nh-7b-qlora-v2"

def download_model() -> tuple:
    """Download the model and tokenizer."""
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    m = PeftModel.from_pretrained(m, adapters_name)
    m = m.merge_and_unload()
    tok = LlamaTokenizer.from_pretrained(model_name)
    return m, tok

if __name__ == "__main__":
    download_model()
    