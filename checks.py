import tempfile
import torch
from model import WindowConfig, WindowLM

torch.set_num_threads(4)
torch.manual_seed(7)
for window in (1, 4, 8, 64):
    m = WindowLM(WindowConfig(vocab_size=32, width=32, layers=3, heads=4, window=window)).eval()
    ids = torch.randint(0, 32, (2, 64))
    with torch.no_grad():
        full = m(ids).logits
        for chunk in (1, 7, 64):
            cache, outputs = None, []
            for start in range(0, 64, chunk):
                out = m(ids[:, start:start+chunk], past_key_values=cache, use_cache=True,
                        position_ids=torch.arange(start, min(start+chunk, 64)))
                cache = out.past_key_values
                outputs.append(out.logits)
                assert all(k.shape[2] <= window-1 for k, v, p in cache)
            torch.testing.assert_close(full, torch.cat(outputs, 1), atol=2e-6, rtol=2e-5)
        changed = ids.clone()
        changed[:, 32:] = (changed[:, 32:] + 1) % 32
        torch.testing.assert_close(full[:, :32], m(changed).logits[:, :32])
    # Last-token gradient must be zero outside the structural receptive field.
    embedding = m.token(ids).detach().requires_grad_(True)
    handle = m.token.register_forward_hook(lambda module, args, output: embedding)
    m(ids).logits[:, -1, 0].sum().backward()
    handle.remove()
    horizon = m.config.layers*(window-1)
    if horizon < 63:
        assert embedding.grad[:, :63-horizon].abs().max() == 0
    if window == 4:
        assert embedding.grad[:, -6].abs().sum() > 0, 'No propagation beyond direct window'
    with tempfile.TemporaryDirectory() as path:
        m.save_pretrained(path, safe_serialization=False)
        restored = WindowLM.from_pretrained(path).eval()
        torch.testing.assert_close(m(ids).logits, restored(ids).logits)
print('PASS: causal masking, full/chunked/token cache equivalence, cache bounds,')
print('      gradient horizon, multi-layer propagation, HF save/load')
