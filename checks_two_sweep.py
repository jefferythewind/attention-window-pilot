"""Independent U/R reference, full gradients, streaming and actual readout path."""
import importlib.util
import subprocess
import tempfile
from pathlib import Path
import torch
from torch.nn import functional as F
from model import WindowConfig,WindowLM
from two_sweep import ordinary_sweep,lateral_sweep,state_bytes

torch.set_num_threads(1);torch.manual_seed(92)

def reference(m,ids):
    positions=torch.arange(ids.shape[1]);x=m.token(ids)+m.position(positions)[None]
    grid=[]
    for b in m.blocks:
        x,_=b(x,positions);grid.append(x)
    b=m.blocks[-1];history=[];out=[];width=m.config.width;heads=m.config.heads
    def split(z):return z.reshape(len(ids),1,heads,width//heads).transpose(1,2)
    for t in range(ids.shape[1]):
        u=grid[-1][:,t:t+1];q,k,v=map(split,b.qkv(b.ln1(u)).chunk(3,-1))
        keys=[];values=[]
        for r in history[-(m.config.lateral_window-1):] if m.config.lateral_window>1 else []:
            _,rk,rv=b.qkv(b.ln1(r)).chunk(3,-1);keys.append(split(rk));values.append(split(rv))
        keys.append(k);values.append(v)
        a=torch.softmax(q@torch.cat(keys,2).transpose(-1,-2)/(width//heads)**.5,-1)@torch.cat(values,2)
        r=u+b.proj(a.transpose(1,2).reshape(len(ids),1,width));history.append(r);out.append(r)
    return m.head(m.norm(torch.cat(out,1))),grid

# Flag-off unchanged against the exact prior architecture source.
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'prior.py';p.write_bytes(subprocess.check_output(['git','show','99f388e:model.py']))
 spec=importlib.util.spec_from_file_location('prior',p);prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
 cfg=dict(vocab_size=16,width=8,heads=2,layers=2,window=3)
 old=prior.WindowLM(prior.WindowConfig(**cfg)).double();new=WindowLM(WindowConfig(**cfg)).double();new.load_state_dict(old.state_dict())
 ids=torch.randint(0,16,(2,8));torch.testing.assert_close(old(ids).logits,new(ids).logits,atol=0,rtol=0)
 old(ids,labels=ids).loss.backward();new(ids,labels=ids).loss.backward()
 for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
for window in (1,3,12):
 for lateral in (1,3,12):
  for length in (1,2,9):
   m=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=window,lateral_window=lateral,two_sweep_recurrent=True)).double().eval()
   with torch.no_grad():
    for b in m.blocks:b.qkv.bias.copy_(torch.linspace(-.2,.1,24))
   ids=torch.randint(0,16,(2,length));gold,grid=reference(m,ids);actual=m(ids).logits
   torch.testing.assert_close(gold,actual,atol=1e-12,rtol=1e-10)
   m.zero_grad();gold.square().sum().backward();grads=[p.grad.clone() for p in m.parameters()]
   m.zero_grad();actual.square().sum().backward()
   for g,p in zip(grads,m.parameters()):torch.testing.assert_close(g,p.grad,atol=1e-10,rtol=1e-8)
   x=m.token(ids)+m.position(torch.arange(length))[None]
   found,_=ordinary_sweep(m,x,torch.arange(length))
   for a,b in zip(grid,found):torch.testing.assert_close(a,b,atol=0,rtol=0)
   # Completed U grid is immutable; ordinary cache must be backbone-only.
   copies=[u.detach().clone() for u in found]
   lateral_sweep(m.blocks[-1],found[-1],torch.arange(length),lateral)
   for a,b in zip(found,copies):torch.testing.assert_close(a,b,atol=0,rtol=0)
   with torch.no_grad():
    for chunk in (1,4,length):
     cache=None;outputs=[]
     for start in range(0,length,chunk):
      o=m(ids[:,start:start+chunk],past_key_values=cache,use_cache=True);cache=o.past_key_values;outputs.append(o.logits)
      seen=start+o.logits.shape[1]
      for s in cache['ordinary']:assert s[0].shape[2]==min(window-1,seen)
      assert cache['recurrent'][0].shape[2]==min(lateral-1,seen)
      assert int(cache['next_position'])==seen
     torch.testing.assert_close(gold,torch.cat(outputs,1),atol=1e-12,rtol=1e-10)
    if length>1:
     changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%16
     torch.testing.assert_close(actual[:,:-1],m(changed).logits[:,:-1],atol=0,rtol=0)
    torch.testing.assert_close(actual,m(ids).logits,atol=0,rtol=0)
    for row in range(2):torch.testing.assert_close(actual[row:row+1],m(ids[row:row+1]).logits,atol=1e-12,rtol=1e-10)
    shifted=m(ids,position_ids=torch.arange(7,7+length)).logits;cache=None;parts=[]
    for start in range(length):
     o=m(ids[:,start:start+1],position_ids=torch.tensor([7]) if start==0 else None,past_key_values=cache,use_cache=True);parts.append(o.logits);cache=o.past_key_values
    torch.testing.assert_close(shifted,torch.cat(parts,1),atol=1e-12,rtol=1e-10)
    try:m(ids[:,:1],past_key_values=cache,position_ids=torch.tensor([0]))
    except ValueError:pass
    else:raise AssertionError('Noncontiguous cached position accepted')
   with torch.no_grad():
    backbone=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=window)).double().eval();backbone.load_state_dict(m.state_dict())
    ordinary_cache=backbone(ids,use_cache=True).past_key_values
    actual_cache=m(ids,use_cache=True).past_key_values['ordinary']
    for a,b in zip(ordinary_cache,actual_cache):
     for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=0,rtol=0)
   assert m.blocks[0].mlp[0].weight.grad.abs().sum()>0
   # Readout explicitly uses corrected TOP state; no lower recurrent rows exist.
   corrected,_=lateral_sweep(m.blocks[-1],found[-1],torch.arange(length),lateral)
   torch.testing.assert_close(actual,m.head(m.norm(corrected)),atol=0,rtol=0)
   assert set(m.state_dict())==set(new.state_dict())
# Controlled history: intervene on completed R-derived cached KV, not U.
b=m.blocks[-1]
with torch.no_grad():
 b.qkv.weight.zero_();b.qkv.bias.zero_();b.qkv.weight[16:]=torch.eye(8)
 b.proj.weight.copy_(torch.eye(8));b.proj.bias.zero_()
u=torch.randn(1,2,8,dtype=torch.double,requires_grad=True)
captured=[]
hook=b.ln1.register_forward_pre_hook(lambda mod,args:captured.append(args[0]))
r,cache=lateral_sweep(b,u[:,:1],torch.tensor([0]),3);hook.remove()
completed=captured[-1]
torch.testing.assert_close(r,completed,atol=0,rtol=0)
changed_r=completed.detach().clone();changed_r[:,:,0]+=.2
kv=F.linear(b.ln1(changed_r),b.qkv.weight[8:],b.qkv.bias[8:]);k,v=kv.reshape(1,1,2,2,4).permute(2,0,3,1,4)
a,_=lateral_sweep(b,u[:,1:],torch.tensor([1]),3,cache)
c,_=lateral_sweep(b,u[:,1:],torch.tensor([1]),3,(k,v,cache[2]))
assert (a-c).abs().max()>1e-8
g=torch.autograd.grad(a[0,0,0],completed,retain_graph=True)[0];assert g.abs().sum()>1e-8
# Beyond ordinary horizon: two layers W2, source distance4 >2.
base=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2)).double().eval()
with torch.no_grad():
 for b in base.blocks:
  b.qkv.weight.zero_();b.qkv.bias.zero_();b.qkv.weight[16:]=torch.eye(8)
  b.proj.weight.copy_(torch.eye(8));b.proj.bias.zero_()
  for p in b.mlp.parameters():p.zero_()
ids=torch.tensor([[1,2,3,4,5]])
for flag in (False,True):
 relay=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2,two_sweep_recurrent=flag)).double().eval();relay.load_state_dict(base.state_dict())
 source=relay.token(ids).detach().requires_grad_(True);h=relay.token.register_forward_hook(lambda mod,args,out:source)
 original=relay(ids).logits[:,-1,0];original.sum().backward();gradient=source.grad[:,0].abs().sum().item();h.remove()
 changed=source.detach().clone();changed[:,0,0]+=.1;h=relay.token.register_forward_hook(lambda mod,args,out:changed)
 influence=(relay(ids).logits[:,-1,0]-original.detach()).abs().item();h.remove()
 assert (gradient>1e-8 and influence>1e-8) if flag else (gradient==0 and influence==0)
 print(f'Two-sweep={flag}: beyond-horizon gradient={gradient:.6g}, influence={influence:.6g}')
with tempfile.TemporaryDirectory() as tmp:
 relay.save_pretrained(tmp,safe_serialization=False);restored=WindowLM.from_pretrained(tmp).double().eval()
 assert restored.config.two_sweep_recurrent
 torch.testing.assert_close(relay(ids).logits,restored(ids).logits,atol=1e-7,rtol=1e-6)
print('PASS: baseline exact; ordinary grid unchanged; independent R-history forward/gradients; top readout; causal/startup/reset/cache/absolute positions; completed-R intervention; beyond-horizon influence/gradient; save/load; no new parameters.')
