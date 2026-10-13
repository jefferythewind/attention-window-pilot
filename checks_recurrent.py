"""Same-layer recurrence: exact specification, causal/reset/cache and gradient checks."""
import importlib.util
import subprocess
import tempfile
from pathlib import Path
import torch
from torch.nn import functional as F
from model import WindowConfig,WindowLM

torch.set_num_threads(1);torch.manual_seed(73)
# Compare flag-off directly to the original published implementation, not itself.
legacy_path=Path(tempfile.gettempdir())/'attention-pilot-original-model.py'
legacy_path.write_bytes(subprocess.check_output(['git','show','ac7dfe0:model.py']))
spec=importlib.util.spec_from_file_location('original_window_model',legacy_path)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
config=dict(vocab_size=16,width=16,heads=2,layers=2,window=3)
a=legacy.WindowLM(legacy.WindowConfig(**config)).double().eval()
b=WindowLM(WindowConfig(**config,recurrent_attention_slot=False)).double().eval();b.load_state_dict(a.state_dict())
ids=torch.randint(0,16,(2,9))
torch.testing.assert_close(a(ids,labels=ids).logits,b(ids,labels=ids).logits,rtol=0,atol=0)
a(ids,labels=ids).loss.backward();b(ids,labels=ids).loss.backward()
for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
assert set(a.state_dict())==set(b.state_dict())

# Independent explicit token/layer reference: compute ordinary and output-derived
# projections, one joint softmax, then both residuals. Nonzero biases are deliberate.
c=WindowLM(WindowConfig(**config,recurrent_attention_slot=True)).double().eval();c.load_state_dict(a.state_dict())
with torch.no_grad():
    for block in c.blocks:block.qkv.bias.copy_(torch.linspace(-.1,.2,48))

def reference(m,ids):
    x=m.token(ids)+m.position(torch.arange(ids.shape[1]))[None]
    histories=[[] for _ in m.blocks];previous=[None for _ in m.blocks];out=[]
    for t in range(ids.shape[1]):
        s=x[:,t:t+1]
        for i,block in enumerate(m.blocks):
            q,k,v=block.qkv(block.ln1(s)).chunk(3,-1)
            split=lambda z:z.reshape(len(ids),1,m.config.heads,-1).transpose(1,2)
            q,k,v=map(split,(q,k,v));histories[i]=histories[i]+[(k,v)]
            local=histories[i][-m.config.window:]
            ks=[p[0] for p in local];vs=[p[1] for p in local]
            if previous[i] is not None:
                _,ek,ev=block.qkv(block.ln1(previous[i])).chunk(3,-1)
                ks.append(split(ek));vs.append(split(ev))
            kk,vv=torch.cat(ks,2),torch.cat(vs,2)
            probs=torch.softmax(q@kk.transpose(-1,-2)/(q.shape[-1]**.5),-1)
            z=s+block.proj((probs@vv).transpose(1,2).reshape(len(ids),1,-1))
            s=z+block.mlp(block.ln2(z));previous[i]=s
        out.append(s)
    return m.head(m.norm(torch.cat(out,1)))
torch.testing.assert_close(c(ids).logits,reference(c,ids),atol=1e-12,rtol=1e-10)
assert sum(p.numel() for p in c.parameters())==sum(p.numel() for p in a.parameters())
c.zero_grad();c(ids).logits.square().sum().backward()
actual_grads={n:p.grad.clone() for n,p in c.named_parameters()}
c.zero_grad();reference(c,ids).square().sum().backward()
for n,p in c.named_parameters():torch.testing.assert_close(actual_grads[n],p.grad,atol=1e-10,rtol=1e-8)


for w in (1,2,3,16):
    torch.manual_seed(91)
    m=WindowLM(WindowConfig(**{**config,'window':w},recurrent_attention_slot=True)).double().eval()
    full=m(ids).logits
    with torch.no_grad():
        for chunk in (1,4,9):
            cache=None;ys=[]
            for start in range(0,ids.shape[1],chunk):
                result=m(ids[:,start:start+chunk],past_key_values=cache,use_cache=True)
                cache=result.past_key_values;ys.append(result.logits)
                for k,v,p,s,nextpos in cache:
                    assert k.shape[2]==min(start+result.logits.shape[1],w-1)
                    assert v.shape==k.shape and p.numel()==k.shape[2]
                    assert s.shape==(2,1,16)
                    assert int(nextpos)==start+result.logits.shape[1]
            torch.testing.assert_close(full,torch.cat(ys,1),atol=1e-12,rtol=1e-10)
        changed=ids.clone();changed[:,5:]=(changed[:,5:]+1)%16
        torch.testing.assert_close(full[:,:5],m(changed).logits[:,:5],atol=0,rtol=0)
        # Answer at last position cannot influence the preceding query.
        changed=ids.clone();changed[:,-1]=(changed[:,-1]+1)%16
        torch.testing.assert_close(full[:,-2],m(changed).logits[:,-2],atol=0,rtol=0)
        # New forward without state resets every layer, including batched examples.
        m(changed,use_cache=True)
        torch.testing.assert_close(full,m(ids).logits,atol=0,rtol=0)
        for row in range(2):torch.testing.assert_close(full[row:row+1],m(ids[row:row+1]).logits,atol=1e-12,rtol=1e-10)
    with tempfile.TemporaryDirectory() as path:
        m.save_pretrained(path,safe_serialization=False)
        restored=WindowLM.from_pretrained(path).double().eval()
        assert restored.config.recurrent_attention_slot
        torch.testing.assert_close(full,restored(ids).logits,atol=1e-7,rtol=1e-6)
    m.train()
    try:m(ids,use_cache=True)
    except ValueError:pass
    else:raise AssertionError('Training accepted a cache')

# Controlled short path: two layers, W=2, distance=4 > horizon=2.
# Uniform attention with identity V/output avoids accidental gradient underflow.
torch.manual_seed(18)
base=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2)).double().eval()
with torch.no_grad():
    for block in base.blocks:
        block.qkv.weight.zero_();block.qkv.bias.zero_();block.qkv.weight[16:]=torch.eye(8)
        block.proj.weight.copy_(torch.eye(8));block.proj.bias.zero_()
        for p in block.mlp.parameters():p.zero_()
small=torch.tensor([[1,2,3,4,5]])
for flag in (False,True):
    m=WindowLM(WindowConfig(vocab_size=16,width=8,heads=2,layers=2,window=2,recurrent_attention_slot=flag)).double().eval()
    m.load_state_dict(base.state_dict())
    emb=m.token(small).detach().requires_grad_(True)
    hook=m.token.register_forward_hook(lambda mod,args,out:emb)
    original=m(small).logits[:,-1,0]
    original.sum().backward();gradient=emb.grad[:,0].abs().sum().item();hook.remove()
    perturb=emb.detach().clone();perturb[:,0,0]+=.1
    hook=m.token.register_forward_hook(lambda mod,args,out:perturb)
    influence=(m(small).logits[:,-1,0]-original.detach()).abs().item();hook.remove()
    if flag:assert gradient>1e-8 and influence>1e-8,(gradient,influence)
    else:assert gradient==0 and influence==0,(gradient,influence)
    print(f'Beyond-horizon recurrence={flag}: gradient={gradient:.6g}, influence={influence:.6g}')
print('PASS: published baseline exact logits/gradients; independent recurrent reference; no new parameters; causality; full/token/chunk consistency; cache bounds; resets; save/load; beyond-horizon forward/gradient path.')
