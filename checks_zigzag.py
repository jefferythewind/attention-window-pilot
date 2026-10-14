"""All-layer vertical integration plus same-row recurrent history and streaming."""
import tempfile
import torch
from model import WindowConfig,WindowLM
from two_sweep import lateral_sweep
from zigzag import grid_sweep

torch.set_num_threads(1);torch.manual_seed(149)

def reference(m,ids,positions=None):
    if positions is None:positions=torch.arange(ids.shape[1])
    x=m.token(ids)+m.position(positions)[None];us=[];rs=[];inputs=[]
    for block in m.blocks:
        inputs.append(x);u,_=block(x,positions);us.append(u)
        history=[];out=[];width=m.config.width;heads=m.config.heads
        def split(z):return z.reshape(len(ids),1,heads,width//heads).transpose(1,2)
        for t in range(ids.shape[1]):
            current=u[:,t:t+1];q,k,v=map(split,block.qkv(block.ln1(current)).chunk(3,-1))
            keys=[];values=[]
            old=history[-(m.config.lateral_window-1):] if m.config.lateral_window>1 else []
            for r in old:
                _,rk,rv=block.qkv(block.ln1(r)).chunk(3,-1);keys.append(split(rk));values.append(split(rv))
            keys.append(k);values.append(v)
            a=torch.softmax(q@torch.cat(keys,2).transpose(-1,-2)/(width//heads)**.5,-1)@torch.cat(values,2)
            r=current+block.proj(a.transpose(1,2).reshape(len(ids),1,width));history.append(r);out.append(r)
        x=torch.cat(out,1);rs.append(x)
    return m.head(m.norm(x)),us,rs,inputs

for window in (1,3,12):
 for lateral in (1,3,12):
  for length in (1,2,9):
   m=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=3,window=window,lateral_window=lateral,zigzag_recurrent=True)).double().eval()
   with torch.no_grad():
    for b in m.blocks:b.qkv.bias.copy_(torch.linspace(-.2,.1,24))
   ids=torch.randint(0,16,(2,length));gold,us,rs,inputs=reference(m,ids);actual=m(ids).logits
   torch.testing.assert_close(gold,actual,atol=1e-12,rtol=1e-10)
   m.zero_grad();gold.square().sum().backward();grads=[p.grad.clone() for p in m.parameters()]
   m.zero_grad();actual.square().sum().backward()
   for g,p in zip(grads,m.parameters()):torch.testing.assert_close(g,p.grad,atol=1e-10,rtol=1e-8)
   embed=m.token(ids)+m.position(torch.arange(length))[None]
   corrected,oc,rc,uu,rr=grid_sweep(m,embed,torch.arange(length),use_cache=True,record_grid=True)
   for a,b in zip(uu,us):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
   for a,b in zip(rr,rs):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
   # Every corrected row has an actual forward/autograd path to the head.
   row_grads=torch.autograd.grad(m.head(m.norm(corrected)).square().sum(),rr,retain_graph=True)
   assert all(g.abs().sum()>1e-8 for g in row_grads)
   for layer in range(3):
    expected,_=m.blocks[layer](embed if layer==0 else rr[layer-1],torch.arange(length),use_cache=True)
    torch.testing.assert_close(uu[layer],expected,atol=0,rtol=0)
   with torch.no_grad():
    for chunk in (1,4,length):
     cache=None;pieces=[]
     for start in range(0,length,chunk):
      o=m(ids[:,start:start+chunk],past_key_values=cache,use_cache=True);cache=o.past_key_values;pieces.append(o.logits)
      seen=start+o.logits.shape[1]
      for s in cache['ordinary']:assert s[0].shape[2]==min(window-1,seen)
      for s in cache['recurrent_layers']:assert s[0].shape[2]==min(lateral-1,seen)
      assert int(cache['next_position'])==seen
     torch.testing.assert_close(gold,torch.cat(pieces,1),atol=1e-12,rtol=1e-10)
    if length>1:
     changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%16
     torch.testing.assert_close(actual[:,:-1],m(changed).logits[:,:-1],atol=0,rtol=0)
    torch.testing.assert_close(actual,m(ids).logits,atol=0,rtol=0)
    for row in range(2):torch.testing.assert_close(actual[row:row+1],m(ids[row:row+1]).logits,atol=1e-12,rtol=1e-10)
    shifted=m(ids,position_ids=torch.arange(7,7+length)).logits;cache=None;pieces=[]
    for start in range(length):
     o=m(ids[:,start:start+1],position_ids=torch.tensor([7]) if start==0 else None,past_key_values=cache,use_cache=True);cache=o.past_key_values;pieces.append(o.logits)
    torch.testing.assert_close(shifted,torch.cat(pieces,1),atol=1e-12,rtol=1e-10)
# Completed same-row history is separate for every layer; intervention is local.
b=m.blocks[0]
with torch.no_grad():
 b.qkv.weight.zero_();b.qkv.bias.zero_();b.qkv.weight[16:]=torch.eye(8)
 b.proj.weight.copy_(torch.eye(8));b.proj.bias.zero_()
u=torch.randn(1,2,8,dtype=torch.double,requires_grad=True);captured=[]
hook=b.ln1.register_forward_pre_hook(lambda mod,args:captured.append(args[0]))
r,cache=lateral_sweep(b,u[:,:1],torch.tensor([0]),3);hook.remove();completed=captured[-1]
a,_=lateral_sweep(b,u[:,1:],torch.tensor([1]),3,cache)
assert torch.autograd.grad(a[0,0,0],completed,retain_graph=True)[0].abs().sum()>1e-8
changed_r=completed.detach().clone();changed_r[:,:,0]+=.2
kv=torch.nn.functional.linear(b.ln1(changed_r),b.qkv.weight[8:],b.qkv.bias[8:]);k,v=kv.reshape(1,1,2,2,4).permute(2,0,3,1,4)
altered,_=lateral_sweep(b,u[:,1:],torch.tensor([1]),3,(k,v,cache[2]));assert (a-altered).abs().max()>1e-8
# Source distance5 exceeds both ordinary horizon2 and U-only two-pass horizon4.
torch.manual_seed(18)
base=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2)).double().eval()
with torch.no_grad():
 for block in base.blocks:
  block.qkv.weight.zero_();block.qkv.bias.zero_();block.qkv.weight[16:]=torch.eye(8)
  block.proj.weight.copy_(torch.eye(8));block.proj.bias.zero_()
  for p in block.mlp.parameters():p.zero_()
ids=torch.tensor([[1,2,3,4,5,6]])
for flag in (False,True):
 relay=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2,zigzag_recurrent=flag)).double().eval();relay.load_state_dict(base.state_dict())
 source=relay.token(ids).detach().requires_grad_(True);h=relay.token.register_forward_hook(lambda mod,args,out:source)
 original=relay(ids).logits[:,-1,0];original.sum().backward();gradient=source.grad[:,0].abs().sum().item();h.remove()
 changed=source.detach().clone();changed[:,0,0]+=.1;h=relay.token.register_forward_hook(lambda mod,args,out:changed)
 influence=(relay(ids).logits[:,-1,0]-original.detach()).abs().item();h.remove()
 assert (gradient>1e-8 and influence>1e-8) if flag else (gradient==0 and influence==0)
 print(f'Zigzag={flag}: beyond-horizon gradient={gradient:.6g}, influence={influence:.6g}')
with tempfile.TemporaryDirectory() as tmp:
 relay.save_pretrained(tmp,safe_serialization=False);restored=WindowLM.from_pretrained(tmp).double().eval()
 assert restored.config.zigzag_recurrent and not restored.config.two_sweep_recurrent
 torch.testing.assert_close(relay(ids).logits,restored(ids).logits,atol=1e-7,rtol=1e-6)
print('PASS: independent all-layer reference outputs/gradients; corrected vertical inputs; every R row reaches head; same-row R history; causal/full/token/chunk/cache/reset/positions; beyond-horizon influence/gradient; save/load.')
