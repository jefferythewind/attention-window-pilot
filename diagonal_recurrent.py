"""Differentiable token/layer wavefronts, batched across independent diagonal nodes."""
import torch
from torch.nn import functional as F


def execute(model, embeddings, positions, past=None):
    cfg=model.config;batch,length,width=embeddings.shape;layers=len(model.blocks)
    heads=cfg.heads;dim=width//heads;window=cfg.window;slots=cfg.recurrent_attention_slots
    def stack(name):
        return torch.stack([dict(block.named_parameters())[name] for block in model.blocks])
    weights={n:stack(n) for n in ('ln1.weight','ln1.bias','qkv.weight','qkv.bias','proj.weight','proj.bias','ln2.weight','ln2.bias','mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias')}
    def select(name,indices):
        return weights[name] if indices is None else weights[name][indices]
    def affine(x,name,indices):
        return torch.bmm(x,select(name+'.weight',indices).transpose(1,2))+select(name+'.bias',indices).unsqueeze(1)
    def normalize(x,name,indices):
        return F.layer_norm(x,(width,),eps=model.blocks[0].ln1.eps)*select(name+'.weight',indices).unsqueeze(1)+select(name+'.bias',indices).unsqueeze(1)
    states=[None]*layers if past is None else list(past)
    previous_output=None;previous_layers=None;batched_state=None;final=[]
    # Node(t,l) depends only on earlier diagonals: (t,l-1), (t-1,l),
    # (t-2,l), and older ordinary KVs. Every node on t+l=d is independent.
    for diagonal in range(length+layers-1):
        ls=list(range(max(0,diagonal-length+1),min(layers-1,diagonal)+1))
        ts=[diagonal-l for l in ls];indices=None if len(ls)==layers else torch.tensor(ls,device=embeddings.device)
        if ls[0]==0:
            first=embeddings[:,diagonal].unsqueeze(0)
            x=first if len(ls)==1 else torch.cat((first,previous_output[:len(ls)-1]),dim=0)
        else:
            x=previous_output[:len(ls)]
        count=len(ls)
        projected=affine(normalize(x,'ln1',indices),'qkv',indices)
        q,k,v=projected.reshape(count,batch,3,heads,dim).permute(2,0,1,3,4)
        prior=[states[l] for l in ls]
        # Fixed padded shapes permit one SDPA call across all diagonal nodes.
        # Missing ordinary/recurrent entries are fully masked, not dummy slots.
        if ls==previous_layers:
            past_k,past_v,history=batched_state
        else:
            zero_k=k.new_zeros(batch,heads,window-1,dim)
            past_k=torch.stack([zero_k if s is None else F.pad(s[0],(0,0,window-1-s[0].shape[2],0)) for s in prior])
            past_v=torch.stack([zero_k if s is None else F.pad(s[1],(0,0,window-1-s[1].shape[2],0)) for s in prior])
            history=torch.stack([x.new_zeros(batch,slots,width) if s is None else F.pad(s[3],(0,0,slots-s[3].shape[1],0)) for s in prior])
        ordinary_k=torch.cat((past_k,k.unsqueeze(3)),dim=3)
        ordinary_v=torch.cat((past_v,v.unsqueeze(3)),dim=3)
        normalized=normalize(history.reshape(count,batch*slots,width),'ln1',indices)
        extra=torch.bmm(normalized,select('qkv.weight',indices)[:,width:].transpose(1,2))+select('qkv.bias',indices)[:,None,width:]
        extra_k,extra_v=extra.reshape(count,batch,slots,2,heads,dim).permute(3,0,1,4,2,5)
        keys=torch.cat((ordinary_k,extra_k),dim=3);values=torch.cat((ordinary_v,extra_v),dim=3)
        mask=torch.tensor([[j>=window-1-(0 if s is None else s[0].shape[2]) for j in range(window)]+[j>=slots-(0 if s is None else s[3].shape[1]) for j in range(slots)] for s in prior],device=x.device,dtype=torch.bool)
        y=F.scaled_dot_product_attention(q.reshape(count*batch,heads,1,dim),keys.reshape(count*batch,heads,window+slots,dim),values.reshape(count*batch,heads,window+slots,dim),attn_mask=mask[:,None,None,None,:].expand(-1,batch,-1,-1,-1).reshape(count*batch,1,1,window+slots))
        y=y.reshape(count,batch,heads,dim).reshape(count,batch,width)
        z=x+affine(y,'proj',indices)
        hidden=affine(normalize(z,'ln2',indices),'mlp.0',indices)
        output=z+affine(F.gelu(hidden),'mlp.2',indices)
        next_history=torch.cat((history,output.unsqueeze(2)),dim=2)[:,:,-slots:]
        updates=[]
        for i,(t,l,s) in enumerate(zip(ts,ls,prior)):
            keep=min(window-1,(0 if s is None else s[0].shape[2])+1)
            kp=positions[t:t+1] if s is None else torch.cat((s[2],positions[t:t+1]))
            history_count=min(slots,(0 if s is None else s[3].shape[1])+1)
            hist=next_history[i,:,-history_count:]
            state=(ordinary_k[i,:,:,-keep:] if keep else ordinary_k[i,:,:,:0],ordinary_v[i,:,:,-keep:] if keep else ordinary_v[i,:,:,:0],kp[-keep:] if keep else kp[:0],hist,positions[t]+1)
            updates.append((l,state))
            if l==layers-1:final.append(output[i])

        for l,state in updates:states[l]=state
        previous_output=output;previous_layers=ls
        batched_state=(ordinary_k[:,:,:,1:],ordinary_v[:,:,:,1:],next_history)
    # Compact returned states; per-layer views must not retain unrelated lanes.
    states=[(s[0].clone(),s[1].clone(),s[2],s[3].clone(),s[4]) for s in states]
    return torch.stack(final,dim=1),states
