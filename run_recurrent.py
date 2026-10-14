"""Preserved V1 references plus 36 exactly paired recurrent-slot runs; resumable."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import resource
import time
import torch
import transformers
from model import WindowConfig, WindowLM
from run import digest, probe_batch
from recurrent_evaluation import evaluate
from chunk_control import chunk_logits, evaluate_chunked, next_token_loss

ROOT=Path(__file__).parent
WORKER={}


def initialize_worker(config, output):
    torch.set_num_threads(config['threads_per_worker'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw=(ROOT/'shakespeare.txt').read_bytes()
    text=raw.decode('utf-8'); chars=sorted(set(text)); lookup={c:i for i,c in enumerate(chars)}
    data=torch.tensor([lookup[c] for c in text]); split=int(len(data)*.9)
    length=config['sequence_length']; vg=torch.Generator().manual_seed(config['validation_seed'])
    starts=torch.randint(0,len(data)-split-length,(32,16),generator=vg)
    text_validation=[torch.stack([data[split+s:split+s+length] for s in row]) for row in starts]
    vg=torch.Generator().manual_seed(config['validation_seed'])
    probe_validation={d:probe_batch(256,vg,d,length,config['distances']) for d in config['distances']}
    WORKER.update(config=config, output=Path(output), chars=chars, train=data[:split],
                  validations={'text':text_validation,'probe':probe_validation})


def train_one(job):
    task,seed,condition=job
    config=WORKER['config']; length=config['sequence_length']; batch=config['batch_size']
    torch.manual_seed(seed)
    base=WindowLM(WindowConfig(vocab_size=len(WORKER['chars']) if task=='text' else 32,
                               width=config['width'],layers=config['layers'],heads=config['heads'],
                               max_positions=config['max_positions']))
    initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    model=WindowLM(WindowConfig(vocab_size=base.config.vocab_size,width=config['width'],
                                layers=config['layers'],heads=config['heads'],window=condition['window'],
                                max_positions=config['max_positions'],
                                recurrent_attention_slot=condition.get('recurrent_attention_slot',False),
                                recurrent_attention_slots=condition.get('recurrent_attention_slots',1),
                                recurrent_execution=condition.get('recurrent_execution','sequential'),
                                two_sweep_recurrent=condition.get('two_sweep_recurrent',False),
                                lateral_window=condition.get('lateral_window',condition['window'])))
    model.load_state_dict(initial); model.train()
    opt=config['optimizer']
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':opt['matrix_weight_decay']},
        {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}],
        lr=opt['peak_lr'],betas=tuple(opt['betas']))
    generator=torch.Generator().manual_seed(seed+config['training_data_seed_offset'])
    steps=config[task+'_steps']; data_hash=hashlib.sha256(); target_hash=hashlib.sha256()
    history=[]; targets=0; start=time.perf_counter()
    for step in range(steps):
        if task=='text':
            train=WORKER['train']; starts=torch.randint(0,len(train)-length,(batch,),generator=generator)
            x=torch.stack([train[s:s+length] for s in starts]); labels=x
        else:
            x,labels,_=probe_batch(batch,generator,sequence_length=length,distances=config['distances'])
        data_hash.update(x.numpy().tobytes()); target_hash.update(labels.numpy().tobytes())
        targets+=int((labels[:,1:]!=-100).sum())
        warmup=min(opt['warmup_steps'],max(1,steps//10))
        scale=(step+1)/warmup if step<warmup else opt['min_lr_fraction']+(1-opt['min_lr_fraction'])*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup-1)))
        for group in optimizer.param_groups:group['lr']=opt['peak_lr']*scale
        optimizer.zero_grad(set_to_none=True)
        if condition['mode']=='chunk':
            loss=next_token_loss(chunk_logits(model,x,condition['window']),labels)
        else:
            loss=model(x,labels=labels).loss
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss in {job}, step {step}')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),opt['grad_clip'])
        optimizer.step()
        if step%50==0 or step==steps-1:
            history.append({'step':step+1,'loss':float(loss.detach())})
            progress=WORKER['output']/f'progress-{task}-seed{seed}-{condition["name"]}.json'
            progress.write_text(json.dumps({'task':task,'seed':seed,'condition':condition['name'],'step':step+1,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-start}))
    elapsed=time.perf_counter()-start
    training_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    validation=WORKER['validations'][task]
    rolling=evaluate(model,task,validation)
    native=None
    if condition['window']==32 and not condition.get('recurrent_attention_slot',False) and not condition.get('two_sweep_recurrent',False):
        native=evaluate_chunked(model,task,validation,32)
        if task=='probe':
            for d in config['distances']:
                if not native[str(d)]['source_in_query_chunk']:
                    assert native[str(d)]['max_source_change_logit_difference']==0
    name=f'{task}-seed{seed}-{condition["name"]}'
    model.save_pretrained(WORKER['output']/name,safe_serialization=False)
    row={'task':task,'seed':seed,'condition':condition['name'],'training_mode':condition['mode'],
         'window':condition['window'],'recurrent_attention_slot':condition.get('recurrent_attention_slot',False),
         'recurrent_attention_slots':condition.get('recurrent_attention_slots',1),
         'two_sweep_recurrent':condition.get('two_sweep_recurrent',False),
         'lateral_window':condition.get('lateral_window',condition['window']),
         'training_peak_process_rss_bytes':training_peak_rss_bytes,
         'recurrent_execution':condition.get('recurrent_execution','sequential'),'parameters':sum(p.numel() for p in model.parameters()),
         'initialization_sha256':digest(initial),'training_sequences_sha256':data_hash.hexdigest(),
         'training_labels_sha256':target_hash.hexdigest(),'processed_tokens':steps*batch*length,
         'supervised_targets':targets,'optimizer_updates':steps,'training_seconds':elapsed,
         'tokens_per_second':steps*batch*length/elapsed,'history':history,'rolling_metrics':rolling,
         'chunked_metrics':native,'checkpoint':name}
    return row


def check_pairs(results, config, complete=False):
    for task in ('text','probe'):
        for seed in config['seeds']:
            rows=[r for r in results if r['task']==task and r['seed']==seed]
            if complete:assert len(rows)==len(config['conditions'])
            if not rows:continue
            assert len({r['condition'] for r in rows})==len(rows)
            for field in ('initialization_sha256','training_sequences_sha256','training_labels_sha256',
                          'processed_tokens','supervised_targets','optimizer_updates','parameters'):
                assert len({r[field] for r in rows})==1,(task,seed,field)
            steps=config[task+'_steps']; batch=config['batch_size']; length=config['sequence_length']
            assert rows[0]['processed_tokens']==steps*batch*length
            assert rows[0]['supervised_targets']==steps*batch*(length-1 if task=='text' else 1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'recurrent_config.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results-recurrent')
    args=parser.parse_args(); config=json.loads(args.config.read_text())
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    raw=(ROOT/'shakespeare.txt').read_bytes(); chars=sorted(set(raw.decode('utf-8')))
    (out/'tokenizer.json').write_text(json.dumps({'type':'character','characters':chars}))
    path=out/'runs.json'; now=time.perf_counter()
    if path.exists():
        metadata=json.loads(path.read_text())
        assert metadata['config']==config,'Existing output uses a different configuration.'
        assert metadata['dataset_sha256']==hashlib.sha256(raw).hexdigest()
    else:
        metadata={'version':config['version'],'config':config,'device':'cpu','dtype':'float32',
                  'torch':torch.__version__,'transformers':transformers.__version__,
                  'python':platform.python_version(),'dataset_sha256':hashlib.sha256(raw).hexdigest(),
                  'dataset_url':'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                  'split_index':int(len(raw.decode('utf-8'))*.9),
                  'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                   for name in ('model.py','run.py','chunk_control.py','run_recurrent.py','recurrent_evaluation.py','recurrent_config.json')},
                  'wall_seconds':0.,'results':[]}
    metadata['prior_wall_seconds']=metadata.get('wall_seconds',0.)
    jobs=[(task,seed,c) for task in ('text','probe') for seed in config['seeds'] for c in config['conditions']]
    reference_path=ROOT/'results-v1/runs.json'
    reference=json.loads(reference_path.read_text())
    if not metadata['results']:
        assert metadata['dataset_sha256']==reference['dataset_sha256']
        for key in ('seeds','sequence_length','layers','width','heads','max_positions','batch_size','text_steps','probe_steps','distances','optimizer','training_data_seed_offset','validation_seed'):
            assert config[key]==reference['config'][key],key
        metadata['results']=[{**r,'recurrent_attention_slot':False,'reused_v1':True} for r in reference['results']]
        metadata['v1_results_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    else:
        assert metadata['v1_results_sha256']==hashlib.sha256(reference_path.read_bytes()).hexdigest()
        for name,expected in metadata['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,('Changed source on resume',name)
    done={(r['task'],r['seed'],r['condition']) for r in metadata['results']}
    for r in metadata['results']:
        directory=ROOT/'results-v1' if r.get('reused_v1') else out
        assert (directory/r['checkpoint']/'pytorch_model.bin').exists()
    check_pairs(metadata['results'],config)
    pending=[job for job in jobs if (job[0],job[1],job[2]['name']) not in done]
    with ProcessPoolExecutor(max_workers=config['workers'],mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker,initargs=(config,str(out))) as executor:
        futures=[executor.submit(train_one,job) for job in pending]
        for future in as_completed(futures):
            row=future.result(); metadata['results'].append(row)
            order={(t,s,c['name']):i for i,(t,s,c) in enumerate(jobs)}
            metadata['results'].sort(key=lambda r:order[r['task'],r['seed'],r['condition']])
            check_pairs(metadata['results'],config)
            metadata['wall_seconds_session']=time.perf_counter()-now
            tmp=out/'runs.tmp.json'; tmp.write_text(json.dumps(metadata,indent=2)); tmp.replace(path)
            metadata['wall_seconds']=metadata.get('prior_wall_seconds',0.)+time.perf_counter()-now
            score=row['rolling_metrics']['heldout_nll'] if row['task']=='text' else row['rolling_metrics'][str(max(config['distances']))]['accuracy']
            print(f'{len(metadata["results"])}/{len(jobs)} {row["task"]} seed={row["seed"]} {row["condition"]} '
                  f'train={row["training_seconds"]:.1f}s rolling_score={score:.4f}',flush=True)
    check_pairs(metadata['results'],config,complete=True)
    metadata['wall_seconds']=metadata['prior_wall_seconds']+time.perf_counter()-now
    path.write_text(json.dumps(metadata,indent=2))
    print('COMPLETE: all paired initialization, input, label, token, target and update checks passed.',flush=True)


if __name__=='__main__':main()
