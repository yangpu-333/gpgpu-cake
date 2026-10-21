"""Read only the two named experiment Pods using existing in-cluster identity.

No credentials are printed, exported, or persisted. Verify the cluster CA and
connect directly to the private API service. Never list other users' resources.
"""
from datetime import datetime, timezone
import ipaddress
import json
import os
from pathlib import Path
import ssl
import socket
import urllib.error
import urllib.request

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
NAMES=('sg-huids25-261017-5453d-default0-0','sg-huids25-261017-3879e-default0-0')


def main():
    account=Path('/var/run/secrets/kubernetes.io/serviceaccount')
    if not os.access(account/'token',os.R_OK):
        result={'schema':'jiuge7b-owned-Pod-readonly-status-v1',
                'utc':datetime.now(timezone.utc).isoformat(),
                'status':'current_UID_cannot_read_incluster_identity; administrator_needed',
                'api_requests_sent':0,'mutating_requests':False,
                'credential_values_read_printed_exported_or_saved':False,
                'permission_changes_or_elevation_attempted':False}
        with (PREP/'sft-preparation-20261017-01/owned-pod-status-readonly.json').open('x') as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps(result),flush=True)
        return
    namespace=(account/'namespace').read_text().strip()
    host=os.environ.get('KUBERNETES_SERVICE_HOST','kubernetes.default.svc')
    port=int(os.environ.get('KUBERNETES_SERVICE_PORT_HTTPS','443'))
    address=ipaddress.ip_address(socket.gethostbyname(host))
    if namespace!='crater-workspace' or not address.is_private or port!=443:
        raise ValueError('Expected existing private ACT cluster API and namespace only')
    context=ssl.create_default_context(cafile=str(account/'ca.crt'))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                      urllib.request.HTTPSHandler(context=context))
    records=[]
    for name in NAMES:
        request=urllib.request.Request(f'https://{host}:{port}/api/v1/namespaces/{namespace}/pods/{name}',
            headers={'Authorization':'Bearer '+(account/'token').read_text().strip()},method='GET')
        try:
            with opener.open(request,timeout=15) as response:
                data=json.load(response)
            metadata=data['metadata']
            record={'name':metadata['name'],'namespace':metadata['namespace'],'uid':metadata['uid'],
                'http_status':200,'deletion_timestamp':metadata.get('deletionTimestamp'),
                'finalizers':metadata.get('finalizers',[]),'owner_references':metadata.get('ownerReferences',[]),
                'own_task_labels':{key:value for key,value in metadata.get('labels',{}).items()
                                   if key in ('crater.raids.io/task-user','crater.raids.io/base-url')},
                'node':data.get('spec',{}).get('nodeName'),'phase':data.get('status',{}).get('phase')}
            records.append(record)
        except urllib.error.HTTPError as error:
            records.append({'name':name,'http_status':error.code,'resource_payload_not_exported':True})
            if error.code in (401,403):
                break
    result={'schema':'jiuge7b-owned-Pod-readonly-status-v1','utc':datetime.now(timezone.utc).isoformat(),
            'namespace':namespace,'records':records,'mutating_requests':False,
            'other_users_resources_read':False,'credential_values_printed_exported_or_saved':False,
            'cluster_CA_verification':True,'HTTP_proxy_disabled_for_incluster_credentials':True}
    with (PREP/'sft-preparation-20261017-01/owned-pod-status-readonly.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
