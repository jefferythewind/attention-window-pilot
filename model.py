import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class WindowConfig(PretrainedConfig):
    model_type = 'window_pilot'

    def __init__(self, vocab_size=64, width=64, layers=4, heads=4,
                 window=64, max_positions=512, recurrent_attention_slot=False,
                 recurrent_attention_slots: int = 1, recurrent_execution='sequential', **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.width, self.layers = vocab_size, width, layers
        self.heads, self.window, self.max_positions = heads, window, max_positions
        self.recurrent_attention_slot = recurrent_attention_slot
        self.recurrent_attention_slots = recurrent_attention_slots
        self.recurrent_execution = recurrent_execution
        if recurrent_execution not in ('sequential', 'diagonal'):
            raise ValueError('recurrent_execution must be sequential or diagonal.')
        if type(recurrent_attention_slots) is not int or recurrent_attention_slots not in (1, 2, 3):
            raise ValueError('recurrent_attention_slots must be 1, 2 or 3.')
        if window < 1 or layers < 1 or width % heads:
            raise ValueError('Positive window/layers and width divisible by heads required.')


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.ln1, self.ln2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3*c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(),
                                 nn.Linear(4*c.width, c.width))

    def forward(self, x, positions, past=None, use_cache=False):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, t, 3, self.c.heads,
                                              d//self.c.heads).permute(2, 0, 3, 1, 4)
        kp = positions
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
            kp = torch.cat((past[2], positions))
        distance = positions[:, None] - kp[None, :]
        mask = (distance >= 0) & (distance < self.c.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        x = x + self.mlp(self.ln2(x))
        cache = None
        if use_cache:
            # W includes the current token; only W-1 past keys are needed.
            keep = self.c.window - 1
            cache = (k[:, :, -keep:] if keep else k[:, :, :0],
                     v[:, :, -keep:] if keep else v[:, :, :0],
                     kp[-keep:] if keep else kp[:0])
        return x, cache

    def recurrent_step(self, x, position, past=None):
        """One token; ordinary input KVs and completed-output state stay separate.

        State is (ordinary K, ordinary V, positions, completed-output history,
        next position). History is oldest first, with at most one, two or three entries.
        No state is detached: training retains the entire temporal graph.
        """
        b, t, d = x.shape
        if t != 1:
            raise ValueError('recurrent_step expects exactly one token.')
        q, k, v = self.qkv(self.ln1(x)).reshape(b, 1, 3, self.c.heads,
                                              d//self.c.heads).permute(2, 0, 3, 1, 4)
        kp = position
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
            kp = torch.cat((past[2], position))
        # Ordinary cache contains W-1 past inputs, so current-token inclusion gives W.
        keys, values = k, v
        if past is not None:
            count = past[3].shape[1]
            kv = F.linear(self.ln1(past[3]), self.qkv.weight[d:], self.qkv.bias[d:])
            extra_k, extra_v = kv.reshape(b, count, 2, self.c.heads,
                                         d//self.c.heads).permute(2, 0, 3, 1, 4)
            keys = torch.cat((keys, extra_k), dim=2)
            values = torch.cat((values, extra_v), dim=2)
        # Every supplied ordinary position is causal and within W. Available
        # recurrent slots share the same softmax; missing history is omitted.
        y = F.scaled_dot_product_attention(q, keys, values)
        z = x + self.proj(y.transpose(1, 2).reshape(b, 1, d))
        completed = z + self.mlp(self.ln2(z))
        keep = self.c.window-1
        history = completed
        if self.c.recurrent_attention_slots > 1 and past is not None:
            history = torch.cat((past[3], completed), dim=1)[:, -self.c.recurrent_attention_slots:]
        state = (k[:, :, -keep:] if keep else k[:, :, :0],
                 v[:, :, -keep:] if keep else v[:, :, :0],
                 kp[-keep:] if keep else kp[:0], history, position[-1]+1)
        return completed, state


class WindowLM(PreTrainedModel):
    config_class = WindowConfig

    def __init__(self, config):
        super().__init__(config)
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_positions, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None, past_key_values=None,
                position_ids=None, use_cache=False, **kwargs):
        if kwargs.get('attention_mask') is not None:
            raise ValueError('This pilot supports unpadded equal-length batches only.')
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError('Training must retain the complete sequence graph.')
        if position_ids is None:
            start = 0
            if past_key_values is not None and self.config.recurrent_attention_slot:
                start = int(past_key_values[0][4])
            elif past_key_values is not None and past_key_values[0][2].numel():
                start = int(past_key_values[0][2][-1]) + 1
            elif past_key_values is not None:
                raise ValueError('Supply absolute position_ids for an empty cache (W=1).')
            position_ids = torch.arange(start, start+input_ids.shape[1], device=input_ids.device)
        if position_ids.ndim == 2:
            # Independent chunks can have different absolute offsets in each row.
            # Attention distances within each row are still ordinary token offsets.
            if use_cache or past_key_values is not None:
                raise ValueError('Batched position offsets support independent chunks without caching.')
            if position_ids.shape != input_ids.shape or not torch.all(position_ids[:, 1:]-position_ids[:, :-1] == 1):
                raise ValueError('Each row needs contiguous absolute positions matching input_ids.')
            x = self.token(input_ids) + self.position(position_ids)
            attention_positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        else:
            x = self.token(input_ids) + self.position(position_ids)[None, :, :]
            attention_positions = position_ids
        if self.config.recurrent_attention_slot:
            if not torch.all(attention_positions[1:]-attention_positions[:-1] == 1):
                raise ValueError('Recurrent sequences need contiguous positions.')
            if past_key_values is not None:
                if len(past_key_values) != len(self.blocks):
                    raise ValueError('Expected one recurrent state per layer.')
                for state in past_key_values:
                    if (len(state) != 5 or state[3].ndim != 3 or
                            state[3].shape[0] != x.shape[0] or
                            state[3].shape[2] != self.config.width or
                            not 1 <= state[3].shape[1] <= self.config.recurrent_attention_slots):
                        raise ValueError('Invalid recurrent state or batch size.')
                    if int(state[4]) != int(attention_positions[0]):
                        raise ValueError('Recurrent chunks must continue at the next position.')
            if self.config.recurrent_execution == 'diagonal':
                from diagonal_recurrent import execute
                x, caches = execute(self, x, attention_positions, past_key_values)
            else:
                states = [None]*len(self.blocks) if past_key_values is None else list(past_key_values)
                outputs = []
                for t in range(x.shape[1]):
                    current = x[:, t:t+1]
                    updated = []
                    for i, block in enumerate(self.blocks):
                        current, state = block.recurrent_step(current, attention_positions[t:t+1], states[i])
                        updated.append(state)
                    outputs.append(current)
                    states = updated
                x = torch.cat(outputs, dim=1)
                caches = states
        else:
            caches = []
            for i, block in enumerate(self.blocks):
                x, cache = block(x, attention_positions,
                                 None if past_key_values is None else past_key_values[i], use_cache)
                caches.append(cache)
        logits = self.head(self.norm(x))
        loss = None if labels is None else F.cross_entropy(
            logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1),
            ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                     past_key_values=tuple(caches) if use_cache else None)
