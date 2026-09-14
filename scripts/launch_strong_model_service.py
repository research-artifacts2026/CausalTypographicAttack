"""Reference launcher for repeating the published local Qwen3.5-27B protocol.

Added for release usability after the original run was registered. The original
registration records the actual service arguments; it does not claim this
launcher was part of the original executed-source snapshot.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile


def launch(checkpoint, binary, output, gpus):
    if not (checkpoint/'model.safetensors.index.json').is_file():
        raise ValueError('Download the official checkpoint before launching')
    selected=[int(i) for i in gpus.split(',')]
    if len(selected)!=2 or len(set(selected))!=2 or min(selected)<0:
        raise ValueError('Select exactly two distinct GPU indices')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',18127))==0:
            raise ValueError('The registered local port is already occupied')
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().splitlines()
    if any(i>=len(used) or int(used[i])>=1000 for i in selected):
        raise ValueError('Selected GPUs are occupied; choose idle GPUs')
    args=[str(binary),'serve',str(checkpoint.resolve()),'--served-model-name','qwen35-27b-cta',
          '--host','127.0.0.1','--port','18127','--tensor-parallel-size','2',
          '--disable-custom-all-reduce','--dtype','bfloat16','--generation-config','vllm',
          '--max-model-len','12288','--max-num-seqs','16','--mm-processor-cache-gb','0',
          '--limit-mm-per-prompt','{"image":1}',
          '--mm-processor-kwargs','{"min_pixels":200704,"max_pixels":602112}',
          '--gpu-memory-utilization','0.88','--no-enable-log-requests','--reasoning-parser','qwen3']
    output.mkdir(parents=True,exist_ok=False)
    cache=Path(tempfile.mkdtemp(prefix='cta27-'))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpus,NCCL_P2P_DISABLE='1',TOKENIZERS_PARALLELISM='false')
    for key,name in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TRITON_CACHE_DIR','triton')]:
        path=cache/name;path.mkdir();env[key]=str(path)
    with (output/'server.log').open('wb') as log:
        proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (output/'launch.json').write_bytes((json.dumps({'pid':proc.pid,'args':args,'gpus':gpus},indent=2)+'\n').encode())
    print('Local service starting; inspect server.log and wait for /health before registering or running.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--vllm-bin',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1')
    args=parser.parse_args();launch(args.checkpoint,args.vllm_bin,args.output,args.gpus)
