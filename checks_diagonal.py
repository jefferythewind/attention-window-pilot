"""Two lateral slots and diagonal scheduling: independent reference and state checks."""
import importlib.util
import subprocess
import tempfile
from pathlib import Path
import torch
from model import WindowConfig,WindowLM

torch.set_num_threads(1);torch.manual_seed(61)
def reference(m,ids):
    x=m.token(ids)+m.position(torch.arange(ids.shape[1]))[None]
    histories=[[] for _ in m.blocks];previous=[[] for _ in m.blocks];out=[]
    for t in range(ids.shape[1]):
        s=x[:,t:t+1]
        for i,block in enumerate(m.blocks):
            q,k,v=block.qkv(block.ln1(s)).chunk(3,-1)
            split=lambda z:z.reshape(len(ids),1,m.config.heads,-1).transpose(1,2)
            q,k,v=map(split,(q,k,v));histories[i]=histories[i]+[(k,v)]
            local=histories[i][-m.config.window:]
            ks=[p[0] for p in local];vs=[p[1] for p in local]
            for old in previous[i][-m.config.recurrent_attention_slots:]:
                _,ek,ev=block.qkv(block.ln1(old)).chunk(3,-1)
                ks.append(split(ek));vs.append(split(ev))
            kk,vv=torch.cat(ks,2),torch.cat(vs,2)
            probs=torch.softmax(q@kk.transpose(-1,-2)/(q.shape[-1]**.5),-1)
            z=s+block.proj((probs@vv).transpose(1,2).reshape(len(ids),1,-1))
            s=z+block.mlp(block.ln2(z));previous[i].append(s)
        out.append(s)
    return m.head(m.norm(torch.cat(out,1)))

old_path=Path(tempfile.gettempdir())/'attention-pilot-one-slot-model.py'
old_path.write_bytes(subprocess.check_output(['git','show','24f1619:model.py']))
spec=importlib.util.spec_from_file_location('old_one_slot',old_path)
old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
legacy_cfg=dict(vocab_size=16,width=16,heads=2,layers=3,window=3,recurrent_attention_slot=True)
old_model=old_module.WindowLM(old_module.WindowConfig(**legacy_cfg)).double()
new_model=WindowLM(WindowConfig(**legacy_cfg)).double();new_model.load_state_dict(old_model.state_dict())
legacy_ids=torch.randint(0,16,(2,9))
torch.testing.assert_close(old_model(legacy_ids).logits,new_model(legacy_ids).logits,atol=0,rtol=0)
old_model(legacy_ids,labels=legacy_ids).loss.backward();new_model(legacy_ids,labels=legacy_ids).loss.backward()
for p,q in zip(old_model.parameters(),new_model.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

for slots in (1,2):
 for window in (1,3,16):
  for length in (1,2,9):
   cfg=dict(vocab_size=16,width=16,heads=2,layers=3,window=window,recurrent_attention_slot=True,recurrent_attention_slots=slots)
   sequential=WindowLM(WindowConfig(**cfg)).double().eval()
   with torch.no_grad():
    for block in sequential.blocks:block.qkv.bias.copy_(torch.linspace(-.1,.2,48))
   diagonal=WindowLM(WindowConfig(**cfg,recurrent_execution='diagonal')).double().eval();diagonal.load_state_dict(sequential.state_dict())
   ids=torch.randint(0,16,(2,length))
   gold=reference(sequential,ids)
   torch.testing.assert_close(gold,sequential(ids).logits,atol=1e-12,rtol=1e-10)
   torch.testing.assert_close(gold,diagonal(ids).logits,atol=1e-12,rtol=1e-10)
   sequential.zero_grad();gold.square().sum().backward()
   diagonal.zero_grad();diagonal(ids).logits.square().sum().backward()
   for p,q in zip(sequential.parameters(),diagonal.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
   with torch.no_grad():
    for chunk in (1,4,length):
     cache=None;outputs=[]
     for start in range(0,length,chunk):
      result=diagonal(ids[:,start:start+chunk],past_key_values=cache,use_cache=True)
      cache=result.past_key_values;outputs.append(result.logits)
      for k,v,p,h,n in cache:
       assert k.shape[2]==min(window-1,start+result.logits.shape[1])
       assert h.shape==(2,min(slots,start+result.logits.shape[1]),16)
     torch.testing.assert_close(gold,torch.cat(outputs,1),atol=1e-12,rtol=1e-10)
    if length>1:
     changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%16
     torch.testing.assert_close(diagonal(ids).logits[:,:-1],diagonal(changed).logits[:,:-1],atol=0,rtol=0)
    diagonal(ids,use_cache=True)
    torch.testing.assert_close(gold,diagonal(ids).logits,atol=1e-12,rtol=1e-10)
    for row in range(2):torch.testing.assert_close(gold[row:row+1],diagonal(ids[row:row+1]).logits,atol=1e-12,rtol=1e-10)
   assert set(sequential.state_dict())==set(diagonal.state_dict())
# Isolate t-2: changing only the second completed state must affect the step.
m=WindowLM(WindowConfig(vocab_size=16,width=8,layers=1,heads=2,window=1,recurrent_attention_slot=True,recurrent_attention_slots=2)).double().eval()
b=m.blocks[0]
with torch.no_grad():
 b.qkv.weight.zero_();b.qkv.bias.zero_();b.qkv.weight[16:]=torch.eye(8)
 b.proj.weight.copy_(torch.eye(8));b.proj.bias.zero_()
 for p in b.mlp.parameters():p.zero_()
old=torch.randn(1,2,8,dtype=torch.double,requires_grad=True)
past=(torch.empty(1,2,0,4,dtype=torch.double),torch.empty(1,2,0,4,dtype=torch.double),torch.empty(0,dtype=torch.long),old,torch.tensor(2))
x=torch.randn(1,1,8,dtype=torch.double)
s,_=b.recurrent_step(x,torch.tensor([2]),past)
s[0,0,0].backward();assert old.grad[:,0].abs().sum()>1e-8
changed=old.detach().clone();changed[:,0,0]+=.1
altered,_=b.recurrent_step(x,torch.tensor([2]),(*past[:3],changed,past[4]))
assert (s.detach()-altered).abs().max()>1e-8
with tempfile.TemporaryDirectory() as path:
 diagonal.save_pretrained(path,safe_serialization=False)
 restored=WindowLM.from_pretrained(path).double().eval()
 assert restored.config.recurrent_attention_slots==2 and restored.config.recurrent_execution=='diagonal'
 torch.testing.assert_close(diagonal(ids).logits,restored(ids).logits,atol=1e-7,rtol=1e-6)
# End-to-end diagonal source path: distance4 > L(W-1)=2.
torch.manual_seed(18)
base=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2)).double().eval()
with torch.no_grad():
 for block in base.blocks:
  block.qkv.weight.zero_();block.qkv.bias.zero_();block.qkv.weight[16:]=torch.eye(8)
  block.proj.weight.copy_(torch.eye(8));block.proj.bias.zero_()
  for p in block.mlp.parameters():p.zero_()
small=torch.tensor([[1,2,3,4,5]])
for enabled in (False,True):
 relay=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2,
     recurrent_attention_slot=enabled,recurrent_attention_slots=2,recurrent_execution='diagonal')).double().eval()
 relay.load_state_dict(base.state_dict())
 source=relay.token(small).detach().requires_grad_(True)
 hook=relay.token.register_forward_hook(lambda mod,args,out:source)
 original=relay(small).logits[:,-1,0]
 original.sum().backward();gradient=source.grad[:,0].abs().sum().item();hook.remove()
 changed=source.detach().clone();changed[:,0,0]+=.1
 hook=relay.token.register_forward_hook(lambda mod,args,out:changed)
 influence=(relay(small).logits[:,-1,0]-original.detach()).abs().item();hook.remove()
 if enabled:assert gradient>1e-8 and influence>1e-8,(gradient,influence)
 else:assert gradient==0 and influence==0,(gradient,influence)
 print(f'Diagonal two-slot beyond-horizon enabled={enabled}: gradient={gradient:.6g}, influence={influence:.6g}')
print('PASS: one/two slots match explicit joint-softmax outputs/gradients; diagonal/full/token/chunk equivalence; missing slots masked; causal/reset/bounds checks; t-2 forward/gradient influence; save/load; unchanged parameter keys.')
