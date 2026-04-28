import torch
import sys
sys.path.insert(0, '/home/nvidia/user/sundongdong/model/Qwen3.5-4B-DFlash')
sys.path.insert(0, '/home/nvidia/user/sundongdong/project/algorithm/dflash/dflash')
from transformers import AutoTokenizer, AutoModelForCausalLM
from dflash import DFlashDraftModel
from model import extract_context_feature, build_target_layer_ids

tok = AutoTokenizer.from_pretrained('/home/nvidia/user/sundongdong/model/Qwen3.5-4B-DFlash')
draft = DFlashDraftModel.from_pretrained('/home/nvidia/user/sundongdong/model/Qwen3.5-4B-DFlash', torch_dtype=torch.float32)
target = AutoModelForCausalLM.from_pretrained('/home/nvidia/user/sundongdong/model/Qwen3-4B', torch_dtype=torch.float32)

prompt = 'The quick brown fox'
input_ids = tok.encode(prompt, return_tensors='pt')

with torch.no_grad():
    out = target(input_ids, output_hidden_states=True)
    first_tok = torch.argmax(out.logits[0, -1]).item()
    print('Target first token:', first_tok, repr(tok.decode([first_tok])))

    target_layer_ids = build_target_layer_ids(32, 5)
    target_hidden = extract_context_feature(out.hidden_states, target_layer_ids)

    block_size = 16
    block_ids = torch.full((1, block_size), 248070, dtype=torch.long)
    block_ids[0, 0] = first_tok

    noise_emb = target.model.embed_tokens(block_ids)
    position_ids = torch.arange(block_size).unsqueeze(0)
    draft_out = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_emb,
        position_ids=position_ids,
        use_cache=False
    )

    draft_logits = target.lm_head(draft_out[:, 1:, :])
    draft_tokens = torch.argmax(draft_logits, dim=-1)
    print('Draft tokens[0:3]:', draft_tokens[0, :3].tolist())
    print('Draft decoded:', [repr(tok.decode([t])) for t in draft_tokens[0, :3].tolist()])
