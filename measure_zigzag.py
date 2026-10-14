"""Profile only the new zigzag checkpoint; keep all earlier profiles unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'results-zigzag/costs.json');args=parser.parse_args()
    if args.checkpoint is None:
        data=json.loads((ROOT/'results-zigzag/runs.json').read_text());row=data['results'][0]
        checkpoint=ROOT/'results-zigzag'/row['checkpoint']
    else:checkpoint=args.checkpoint
    result=json.loads(subprocess.check_output([sys.executable,str(ROOT/'measure_two_sweep.py'),'--checkpoint',str(checkpoint)],text=True))
    result['condition']='zigzag-all-layer-recurrent-w32'
    data={'method':'Only the new saved checkpoint is profiled, in a fresh two-thread process. One fixed batch16 x 320 forward/backward with no optimizer update; two warmups, five prefills and twenty frozen-prefix one-token decodes. Parameter values remain bitwise unchanged. Row volumes use diagnostic grid instrumentation, not inference peak allocation.',
          'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('measure_zigzag.py','measure_two_sweep.py','model.py','zigzag.py','two_sweep.py','recurrent_evaluation.py')},
          'hardware':{'platform':platform.platform(),'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_limit_bytes':Path('/sys/fs/cgroup/memory.max').read_text().strip()},'results':[result]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')
    print('Zigzag fixed-batch forward/backward:',result['training_forward_backward_seconds'],'seconds')
    for b,m in result['latencies_and_state_by_batch'].items():print('batch',b,'prefill',m['prefill_median_seconds'],'decode',m['decode_median_seconds'])

if __name__=='__main__':main()
