import torch
from model import WindowConfig, WindowLM
from chunk_control import chunk_logits, next_token_loss

torch.set_num_threads(1)
torch.manual_seed(123)
model=WindowLM(WindowConfig(vocab_size=32,width=32,layers=4,heads=4,window=16)).eval()
x=torch.randint(0,32,(2,64))
batched=chunk_logits(model,x,16)
manual=torch.cat([model(x[:,s:s+16],position_ids=torch.arange(s,s+16)).logits for s in range(0,64,16)],1)
torch.testing.assert_close(batched,manual,atol=2e-6,rtol=2e-5)
changed=x.clone(); changed[:,:48]=(changed[:,:48]+1)%32
torch.testing.assert_close(batched[:,48:],chunk_logits(model,changed,16)[:,48:],atol=0,rtol=0)
future=x.clone(); future[:,63]=(future[:,63]+1)%32
torch.testing.assert_close(batched[:,62],chunk_logits(model,future,16)[:,62],atol=0,rtol=0)

# Chunking into one full block must preserve both shifted loss and gradients.
model.config.window=64
model.train(); model.zero_grad()
loss=model(x,labels=x).loss; loss.backward()
grads={n:p.grad.clone() for n,p in model.named_parameters()}
model.zero_grad(); full_chunk_loss=next_token_loss(chunk_logits(model,x,64),x)
full_chunk_loss.backward()
torch.testing.assert_close(loss,full_chunk_loss)
for n,p in model.named_parameters():torch.testing.assert_close(grads[n],p.grad)

# Every one of the 63 original shifted targets is retained, including boundaries.
logits=torch.randn(2,64,32,requires_grad=True)
next_token_loss(logits,x).backward()
assert torch.all(logits.grad[:,:63].abs().sum(-1)>0)
assert torch.all(logits.grad[:,63]==0)
print('PASS: batched chunks match manual independent forwards; no cross-chunk influence;')
print('      no answer leakage; full-block loss/gradient equivalence; all targets retained.')
