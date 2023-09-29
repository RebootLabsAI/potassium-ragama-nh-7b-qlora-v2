from potassium import Potassium, Request, Response
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

model_name = "NousResearch/Nous-Hermes-llama-2-7b"
adapters_name = "rebootai/ragama-nh-7b-qlora-v2"

app = Potassium("ragama-nh-7b-qlora-v2")

@app.init
def init() -> dict:
    """Initialize the application with the model and tokenizer."""
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    m = PeftModel.from_pretrained(m, adapters_name)
    m = m.merge_and_unload()
    tok = LlamaTokenizer.from_pretrained(model_name)
    tok.bos_token_id = 1

    return {
        "model": m,
        "tokenizer": tok
    }
    
@app.handler()
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate text from a prompt."""
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    max_new_tokens = request.json.get("max_new_tokens", 512)
    temperature = request.json.get("temperature", 0.7)
    prompt = request.json.get("prompt")
    prompt_template=f'''### {prompt} ### Response: \n'''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(output[0])
    return Response(json={"outputs": result}, status=200)

if __name__ == "__main__":
    app.serve()