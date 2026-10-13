"""V1-equivalent evaluation, augmented with recurrent state and source sensitivity."""
import math
import numpy as np
import torch


def state_bytes(cache):
    kv=sum(s[0].numel()*s[0].element_size()+s[1].numel()*s[1].element_size() for s in cache)
    recurrent=sum(s[3].numel()*s[3].element_size() for s in cache if len(s)==5)
    metadata=sum(s[2].numel()*s[2].element_size()+(s[4].numel()*s[4].element_size() if len(s)==5 else 0) for s in cache)
    storages={tensor.untyped_storage().data_ptr():tensor.untyped_storage().nbytes() for s in cache for tensor in s}
    return {'cache_kv_bytes':kv,'cache_recurrent_output_bytes':recurrent,'cache_metadata_bytes':metadata,
            'cache_total_logical_bytes':kv+recurrent+metadata,'cache_backing_storage_bytes':sum(storages.values())}


@torch.no_grad()
def evaluate(model,task,validation):
    model.eval()
    if task=='text':
        losses=[float(model(x,labels=x).loss) for x in validation]
        return {'heldout_nll':float(np.mean(losses)),'perplexity':math.exp(float(np.mean(losses)))}
    result={}
    for distance,(x,labels,source) in validation.items():
        prompt,answer=x[:,:-1],x[:,-1];length=prompt.shape[1]
        output=model(prompt).logits[:,-1];cache=None
        for start in range(0,length,16):
            cached=model(prompt[:,start:start+16],past_key_values=cache,use_cache=True,
                         position_ids=torch.arange(start,min(start+16,length)))
            cache=cached.past_key_values
        torch.testing.assert_close(output,cached.logits[:,-1],atol=3e-6,rtol=3e-5)
        w=model.config.window;recent=prompt[:,-w:]
        recomputed=model(recent,position_ids=torch.arange(length-recent.shape[1],length)).logits[:,-1]
        corrupted=prompt.clone();corrupted[torch.arange(len(x)),source]=(answer+1)%8
        altered=model(corrupted).logits[:,-1]
        result[str(distance)]={'accuracy':float((output.argmax(-1)==answer).float().mean()),
            'answer_nll':float(torch.nn.functional.cross_entropy(output,answer)),
            'recomputed_window_accuracy':float((recomputed.argmax(-1)==answer).float().mean()),
            'counterfactual_accuracy':float((altered.argmax(-1)==(answer+1)%8).float().mean()),
            'prediction_changed_fraction':float((output.argmax(-1)!=altered.argmax(-1)).float().mean()),
            'max_changed_source_logit_difference':float((output-altered).abs().max()),
            'mean_changed_source_logit_difference':float((output-altered).abs().mean()),
            **state_bytes(cache),'cache_batch_size':len(x)}
    return result
