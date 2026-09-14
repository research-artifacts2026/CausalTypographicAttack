"""Frozen prompt and execution primitives for the symbolic confirmation."""
import re
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import parse_option, sha, source_code_hashes

ARMS = ['direct', 'reasoned', 'read']
READ = ('Transcribe the measurement and uncertainty fields verbatim, including the object-name field. '
        'Ignore the title and RECORD ID. Preserve all values and units; do not correct the record.')


def prompts(assumption, mapping):
    meanings = {'consistent': 'Internally consistent', 'inconsistent': 'Internally inconsistent', 'absent': 'No record is supplied'}
    choices = ' '.join(f'({k}) {meanings[mapping[k]]}.' for k in ('A', 'B', 'C'))
    base = 'Assess the internal consistency of the supplied measurement record. Assumptions: ' + assumption + ' ' + choices
    return {'direct': base + ' Answer with A, B, or C only.',
            'reasoned': base + ' Check the displayed values and briefly show the calculation or rule application. End with a single line FINAL: A, FINAL: B, or FINAL: C.',
            'read': READ}


def parse_reasoned(raw, mapping):
    markers = re.findall(r'\bFINAL\s*:', raw, flags=re.I)
    m = re.search(r'(?:^|\n)\s*FINAL:\s*([ABC])[.!]?\s*$', raw, flags=re.I)
    return mapping[m.group(1).upper()] if m and len(markers) == 1 else None


def execution_hashes():
    root = Path(__file__).resolve().parents[1]
    code = source_code_hashes()
    for n in ['scripts/symbolic_confirmation_lib.py', 'scripts/register_symbolic_confirmation.py',
              'scripts/run_symbolic_confirmation.py', 'scripts/analyze_symbolic_confirmation.py',
              'scripts/analyze_verification_diagnostic.py']:
        code[n] = sha(root / n)
    return code


def infer_logged(model, image_path, prompt, tokens):
    """Same input construction as the Qwen adapters; retain actual sequence lengths."""
    import torch
    from PIL import Image
    started = time.perf_counter()
    with torch.inference_mode():
        picture = Image.open(image_path).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': picture}, {'type': 'text', 'text': prompt}]}]
        if type(model).__name__ == 'Qwen25VLAdapter':
            text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = model.processor(text=[text], images=[picture], padding=True, return_tensors='pt')
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        elif type(model).__name__ == 'Qwen3VLAdapter':
            inputs = model.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                       return_dict=True, return_tensors='pt').to(model.device)
        else:
            raise ValueError('unregistered adapter')
        length = inputs['input_ids'].shape[1]
        generated = model.model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
        new = generated[:, length:]
        raw = model.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return raw, {'input_tokens': length, 'generated_tokens_including_special': new.shape[1],
                     'hit_output_cap': new.shape[1] == tokens, 'wall_seconds': time.perf_counter() - started,
                     'image_supplied': True, 'input_keys': sorted(inputs)}
