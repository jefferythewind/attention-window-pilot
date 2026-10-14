"""All-layer ordinary/lateral zigzag; corrected rows feed the next ordinary block."""
import torch
from two_sweep import lateral_sweep


def grid_sweep(model, embeddings, positions, past=None, use_cache=False, record_grid=False):
    ordinary=[]; recurrent=[]; u_grid=[]; r_grid=[]; x=embeddings
    for layer, block in enumerate(model.blocks):
        # The input x is R[layer-1], never an uncorrected previous-layer U.
        u, cache=block(x,positions,None if past is None else past['ordinary'][layer],use_cache)
        x, r_cache=lateral_sweep(block,u,positions,model.config.lateral_window,
                                 None if past is None else past['recurrent_layers'][layer])
        if record_grid:u_grid.append(u);r_grid.append(x)
        if use_cache:
            ordinary.append(tuple(t.clone() for t in cache))
            recurrent.append(tuple(t.clone() for t in r_cache))
    return x,tuple(ordinary),tuple(recurrent),u_grid,r_grid


def execute(model, embeddings, positions, past=None, use_cache=False):
    if not torch.all(positions[1:]-positions[:-1]==1):
        raise ValueError('Zigzag sequences require contiguous absolute positions.')
    if past is not None:
        if set(past)!={'ordinary','recurrent_layers','next_position'}:
            raise ValueError('Invalid zigzag cache format.')
        if len(past['ordinary'])!=len(model.blocks) or len(past['recurrent_layers'])!=len(model.blocks):
            raise ValueError('One ordinary and one recurrent cache required per layer.')
        if int(past['next_position'])!=int(positions[0]):
            raise ValueError('Zigzag chunks must continue at the next absolute position.')
        for states,window in [(past['ordinary'],model.config.window),(past['recurrent_layers'],model.config.lateral_window)]:
            for s in states:
                if s[0].shape[0]!=embeddings.shape[0] or s[0].shape[2]>window-1:
                    raise ValueError('Invalid cache batch or window bound.')
    corrected,ordinary,recurrent,_,_=grid_sweep(model,embeddings,positions,past,use_cache)
    cache={'ordinary':ordinary,'recurrent_layers':recurrent,'next_position':positions[-1]+1} if use_cache else None
    return corrected,cache


def state_bytes(cache):
    ordinary=sum(t.numel()*t.element_size() for s in cache['ordinary'] for t in s[:2])
    recurrent=sum(t.numel()*t.element_size() for s in cache['recurrent_layers'] for t in s[:2])
    metadata=sum(s[2].numel()*s[2].element_size() for s in cache['ordinary']+cache['recurrent_layers'])
    metadata+=cache['next_position'].numel()*cache['next_position'].element_size()
    tensors=[t for s in cache['ordinary']+cache['recurrent_layers'] for t in s]+[cache['next_position']]
    storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
    return {'cache_kv_bytes':ordinary,'cache_recurrent_kv_bytes':recurrent,'cache_metadata_bytes':metadata,
            'cache_total_logical_bytes':ordinary+recurrent+metadata,'cache_backing_storage_bytes':sum(storage.values())}
