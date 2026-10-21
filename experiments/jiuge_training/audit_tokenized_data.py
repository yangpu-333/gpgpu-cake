"""Read-only audit of the native indexed CPT corpus and its document endings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):
            value.update(block)
    return value.hexdigest()


def main():
    sys.path.insert(0,str(PREP/'megatron-native-5be9626'))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    prefix=PREP/'datasets/alpaca-cpt-v1_text_document'
    dataset=IndexedDataset(str(prefix),multimodal=False,mmap=True)
    bad_endings=[]
    total=0
    for index in range(len(dataset)):
        values=dataset[index]
        total+=len(values)
        if not len(values) or int(values[-1])!=73440:
            bad_endings.append(index)
    record={'schema':'jiuge7b-native-indexed-cpt-audit-v1','utc':datetime.now(timezone.utc).isoformat(),
            'documents':len(dataset),'tokens_including_BOS_EOD':total,'EOD':73440,
            'incorrect_document_endings':bad_endings,'source_documents':48818,
            'files_sha256':{suffix:sha(Path(str(prefix)+suffix)) for suffix in ('.bin','.idx')},
            'passed':len(dataset)==48818 and not bad_endings,'gpu_math_or_training':False}
    output=PREP/'datasets/audit.json'
    with output.open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
