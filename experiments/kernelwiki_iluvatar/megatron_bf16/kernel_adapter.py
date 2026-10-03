"""Opt-in CE adapter; true BF16 model weights and native residual/RMSNorm."""
import hashlib
import importlib.util
from pathlib import Path
import types


class Handle:
    def __init__(self,model,candidate,contract):
        self.model=model
        self.contract=contract
        expected=contract['candidate_sha256']
        actual=hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
        if actual!=expected:
            raise RuntimeError('candidate hash differs from sealed BF16 contract')
        spec=importlib.util.spec_from_file_location('bf16_verified_candidate',candidate)
        self.candidate=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.candidate)
        self.counts={'ce_candidate_calls':0,'ce_native_calls':0,'residual_native_calls':0,'fallback_reasons':{}}
        self.observations=[]
        self.devices={}
        self.world=None
        self.config={name:getattr(model.config,name,None) for name in
            ('num_layers','hidden_size','num_attention_heads','ffn_hidden_size','bf16','fp16','normalization',
             'layernorm_epsilon','hidden_dropout','attention_dropout','add_bias_linear',
             'tensor_model_parallel_size','pipeline_model_parallel_size','context_parallel_size')}
        original=model.compute_language_model_loss
        self.original_loss=original
        self.original_bda=[]
        def loss(instance,labels,logits):
            import torch
            import torch.distributed as dist
            self.world=dist.get_world_size()
            signature=contract['model']
            shape=(signature['sequence_length'],signature['micro_batch_size'],signature['vocab_size'])
            reason=None
            if self.world!=1:
                reason='world size'
            elif not logits.is_cuda or logits.device.index!=0:
                reason='device'
            elif torch.is_autocast_enabled():
                reason='autocast'
            elif logits.dtype!=torch.bfloat16 or not logits.is_contiguous() or tuple(logits.shape)!=shape:
                reason='logits dtype/layout/shape'
            elif labels.dtype!=torch.int64 or tuple(labels.shape)!=(shape[1],shape[0]):
                reason='labels dtype/shape'
            else:
                self.devices['0']=torch.cuda.get_device_name(0)
                if self.devices['0'] not in ('BI-V150','Iluvatar BI-V150'):
                    reason='GPU architecture'
            if reason:
                self.counts['ce_native_calls']+=1
                self.counts['fallback_reasons'][reason]=self.counts['fallback_reasons'].get(reason,0)+1
                return original(labels,logits)
            target=labels.transpose(0,1).contiguous()
            result=self.candidate.cross_entropy(logits,target)
            self.counts['ce_candidate_calls']+=1
            if len(self.observations)<4:
                self.observations.append({'logits_shape':list(logits.shape),'logits_dtype':str(logits.dtype),
                    'loss_shape':list(result.shape),'loss_dtype':str(result.dtype),'autocast':False})
            return result.transpose(0,1).contiguous()
        model.compute_language_model_loss=types.MethodType(loss,model)
        for layer in model.decoder.layers:
            original_factory=layer.self_attn_bda
            self.original_bda.append((layer,original_factory))
            def factory(*args,_original=original_factory,**kwargs):
                native=_original(*args,**kwargs)
                def call(*positional,**named):
                    self.counts['residual_native_calls']+=1
                    return native(*positional,**named)
                return call
            layer.self_attn_bda=factory
    def snapshot(self):
        return {'enabled':True,'active':True,'provenance_verified':True,
            'candidate_sha256':self.contract['candidate_sha256'],'candidate_expected_sha256':self.contract['candidate_sha256'],
            'runtime_world_size':self.world,'verified_devices':self.devices,'config_at_installation':self.config,
            'counts':self.counts,'observations':self.observations,
            'scope':'CE-only candidate API; residual/RMSNorm remain native. Counts are not compiled-launch proof.'}
    def uninstall(self):
        self.model.compute_language_model_loss=self.original_loss
        for layer,value in self.original_bda:
            layer.self_attn_bda=value


def install_kernels(model,candidate,contract,megatron_root):
    cfg=model.config
    signature=contract['model']
    for name in ('num_layers','hidden_size','ffn_hidden_size','num_attention_heads'):
        if getattr(cfg,name)!=signature[name]:
            raise RuntimeError('unverified BF16 model configuration: '+name)
    for name,value in {'bf16':True,'fp16':False,'normalization':'RMSNorm','layernorm_epsilon':1e-5,
        'hidden_dropout':0.0,'attention_dropout':0.0,'add_bias_linear':False,
        'tensor_model_parallel_size':1,'pipeline_model_parallel_size':1,'context_parallel_size':1}.items():
        if getattr(cfg,name)!=value:
            raise RuntimeError('unsupported BF16 model setting: '+name)
    if model.vocab_size!=signature['vocab_size'] or len(model.decoder.layers)!=signature['num_layers']:
        raise RuntimeError('unverified vocabulary/layers')
    return Handle(model,candidate,contract)
