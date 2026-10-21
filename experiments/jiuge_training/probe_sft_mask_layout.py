"""CPU interface probe: real indexed documents through native and SFT getitem.

Explicit hand-built index boundaries exercise single/multiple documents and
padding. This is not a native-sampler or model-training acceptance result.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    os.environ['USE_TF']='0'
    sys.path.insert(0,str(PREP/'megatron-native-5be9626'))
    import numpy as np
    import torch
    from megatron.core.datasets.gpt_dataset import GPTDataset
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from sft_dataset_adapter import install
    adapter=install()
    data=IndexedDataset(str(PREP/'datasets/alpaca-sft-answer-v1'),multimodal=False,mmap=True)
    masks=IndexedDataset(str(PREP/'datasets/alpaca-sft-answer-v1_loss_mask'),multimodal=False,mmap=True)
    tests=[]
    for name,length,padding in (('single_document',64,False),('multiple_documents',2048,False),('padded_document',256,True)):
        needed=length+1
        documents=[0]
        if padding:
            end=len(data[0])-1
        else:
            remaining=needed
            while remaining>len(data[documents[-1]]):
                remaining-=len(data[documents[-1]])
                documents.append(documents[-1]+1)
            end=remaining-1
        obj=object.__new__(adapter)
        obj.dataset=data
        obj.answer_masks=masks
        obj.admission_samples=[]
        obj._pad_token_id=-1
        obj.config=SimpleNamespace(add_extra_token_to_sequence=True,reset_position_ids=False,
            reset_attention_mask=False,eod_mask_loss=False,create_attention_mask=True,
            tokenizer=SimpleNamespace(eod=73440),inter_document_masking=False,sequence_length=length)
        obj.shuffle_index=np.array([0],dtype=np.uint32)
        obj.document_index=np.array(documents,dtype=np.int32)
        obj.sample_index=np.array([[0,0],[len(documents)-1,end]],dtype=np.int64)
        obj.masks_and_position_ids_are_cacheable=True
        obj.masks_and_position_ids_are_cached=False
        native=GPTDataset.__getitem__(obj,0)
        adapted=obj[0]
        stream=[]
        for doc in documents:
            stream.extend(masks[doc].tolist())
        stream=stream[:needed]+[0]*max(0,needed-len(stream))
        expected=torch.tensor(stream[1:],dtype=torch.float32)
        if not torch.equal(adapted['loss_mask'],expected):
            raise ValueError(name+' supervision shifted/padded incorrectly')
        for key in ('tokens','labels','position_ids','attention_mask'):
            if not torch.equal(native[key],adapted[key]):
                raise ValueError(name+' changed a native field: '+key)
        repeated=GPTDataset.__getitem__(obj,0)
        if not torch.equal(repeated['loss_mask'],native['loss_mask']):
            raise ValueError('SFT adaptation mutated the native cached loss mask')
        tests.append({'case':name,'sequence_length':length,'documents':documents,
                      'supervised_targets':int(expected.sum()),'mask_shift_and_padding_exact':True,
                      'native_tokens_labels_attention_positions_unchanged':True,
                      'native_cached_loss_mask_unchanged':True})
    result={'schema':'jiuge7b-sft-real-data-layout-probe-v1','passed':True,'checks':tests,
            'cuda_initialized':torch.cuda.is_initialized(),'native_sampler_tested':False,
            'real_model_training_tested':False}
    with (PREP/'sft-preparation-20261017-01/layout-probe.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
