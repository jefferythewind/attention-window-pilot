"""Ordinary parallel backbone, then a temporally recurrent top-row correction.

Lower recurrent rows are intentionally absent: with the existing top-only head
and no vertical integration, their computed states would have no loss path.
"""
import torch
from torch.nn import functional as F


def ordinary_sweep(model, embeddings, positions, past=None, use_cache=False):
    grid=[]; caches=[]; x=embeddings
    for layer, block in enumerate(model.blocks):
        x, cache=block(x, positions, None if past is None else past[layer], use_cache)
        grid.append(x)
        # Prefill slices otherwise retain whole-sequence K/V allocations.
        if cache is not None: cache=tuple(t.clone() for t in cache)
        caches.append(cache)
    return grid, tuple(caches)


def lateral_sweep(block, ordinary_top, positions, window, past=None):
    """Each completed R is projected into history K/V exactly once, without detach."""
    batch,length,width=ordinary_top.shape;heads=block.c.heads;dim=width//heads
    q,k,v=block.qkv(block.ln1(ordinary_top)).reshape(batch,length,3,heads,dim).permute(2,0,3,1,4)
    history=past;outputs=[]
    for t in range(length):
        keys=k[:,:,t:t+1];values=v[:,:,t:t+1]
        if history is not None:
            keys=torch.cat((history[0],keys),dim=2)
            values=torch.cat((history[1],values),dim=2)
        attention=F.scaled_dot_product_attention(q[:,:,t:t+1],keys,values)
        completed=ordinary_top[:,t:t+1]+block.proj(attention.transpose(1,2).reshape(batch,1,width))
        outputs.append(completed)
        kv=F.linear(block.ln1(completed),block.qkv.weight[width:],block.qkv.bias[width:])
        rk,rv=kv.reshape(batch,1,2,heads,dim).permute(2,0,3,1,4)
        kp=positions[t:t+1]
        if history is not None:
            rk=torch.cat((history[0],rk),dim=2);rv=torch.cat((history[1],rv),dim=2)
            kp=torch.cat((history[2],kp))
        keep=window-1
        history=(rk[:,:,-keep:] if keep else rk[:,:,:0],
                 rv[:,:,-keep:] if keep else rv[:,:,:0],kp[-keep:] if keep else kp[:0])
    return torch.cat(outputs,dim=1),history


def execute(model, embeddings, positions, past=None, use_cache=False):
    if not torch.all(positions[1:]-positions[:-1]==1):
        raise ValueError('Two-sweep sequences require contiguous absolute positions.')
    if past is not None:
        if set(past)!= {'ordinary','recurrent','next_position'} or len(past['ordinary'])!=len(model.blocks):
            raise ValueError('Invalid two-sweep cache.')
        if int(past['next_position'])!=int(positions[0]):
            raise ValueError('Two-sweep chunks must continue at the next absolute position.')
        for state,window in [(s,model.config.window) for s in past['ordinary']]+[(past['recurrent'],model.config.lateral_window)]:
            if state[0].shape[0]!=embeddings.shape[0] or state[0].shape[2]>window-1:
                raise ValueError('Invalid batch size or cache bound.')
    grid,ordinary=ordinary_sweep(model,embeddings,positions,None if past is None else past['ordinary'],use_cache)
    corrected,recurrent=lateral_sweep(model.blocks[-1],grid[-1],positions,model.config.lateral_window,
                                       None if past is None else past['recurrent'])
    cache=None
    if use_cache:
        cache={'ordinary':ordinary,'recurrent':tuple(t.clone() for t in recurrent),'next_position':positions[-1]+1}
    return corrected,cache


def state_bytes(cache):
    ordinary=sum(t.numel()*t.element_size() for s in cache['ordinary'] for t in s[:2])
    recurrent=sum(t.numel()*t.element_size() for t in cache['recurrent'][:2])
    metadata=sum(s[2].numel()*s[2].element_size() for s in cache['ordinary'])
    metadata+=cache['recurrent'][2].numel()*cache['recurrent'][2].element_size()+cache['next_position'].numel()*cache['next_position'].element_size()
    tensors=[t for s in cache['ordinary'] for t in s]+list(cache['recurrent'])+[cache['next_position']]
    storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
    return {'cache_kv_bytes':ordinary,'cache_recurrent_kv_bytes':recurrent,'cache_metadata_bytes':metadata,
            'cache_total_logical_bytes':ordinary+recurrent+metadata,'cache_backing_storage_bytes':sum(storage.values())}
