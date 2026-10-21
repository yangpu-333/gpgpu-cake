import unittest

import torch
from jiuge7b_bridge import shard_mapping


class BridgeMappingTests(unittest.TestCase):
    def test_qkv_head_interleave_and_swiglu_tp_stride(self):
        h, f, v = 4, 8, 8
        weights = {'model.embed_tokens.weight': torch.arange(v*h).reshape(v,h),
                   'lm_head.weight': torch.arange(v*h).reshape(v,h), 'model.norm.weight': torch.ones(h)}
        p = 'model.layers.0.'
        weights[p+'input_layernorm.weight'] = weights[p+'post_attention_layernorm.weight'] = torch.ones(h)
        for ordinal, part in enumerate(('q','k','v','o')):
            weights[p+'self_attn.'+part+'_proj.weight'] = torch.arange(h*h).reshape(h,h)+ordinal*100
            if part != 'o':
                weights[p+'self_attn.'+part+'_proj.bias'] = torch.arange(h)+ordinal*100
        weights[p+'mlp.gate_proj.weight'] = torch.arange(f*h).reshape(f,h)
        weights[p+'mlp.up_proj.weight'] = weights[p+'mlp.gate_proj.weight']+1000
        weights[p+'mlp.down_proj.weight'] = torch.arange(h*f).reshape(h,f)
        for rank in range(2):
            result = {name: tensor for name,tensor,_ in shard_mapping(weights,rank,2,1,h,2,f,v)}
            prefix = 'decoder.layers.0.'
            expected = torch.cat([weights[p+'self_attn.'+part+'_proj.weight'][rank*2:(rank+1)*2] for part in ('q','k','v')])
            self.assertTrue(torch.equal(result[prefix+'self_attention.linear_qkv.weight'],expected))
            expected_fc = torch.cat([weights[p+'mlp.'+part+'_proj.weight'][rank*4:(rank+1)*4] for part in ('gate','up')])
            self.assertTrue(torch.equal(result[prefix+'mlp.linear_fc1.weight'],expected_fc))


if __name__ == '__main__':
    unittest.main()
