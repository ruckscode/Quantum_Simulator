"""Audit committed chunks in a paused full-development QMutBench binary run."""
from __future__ import annotations
import json, sys, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qmutbench_qml.training import run_binary_development as checkpointing

def main():
    output=ROOT/'reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029'
    progress_path=output/'full_campaign_manifest.json'
    progress=json.loads(progress_path.read_text(encoding='utf-8'))
    fold=int(progress['active_fold']); chunk_dir=output/'fold_checkpoints'
    manifest_path=chunk_dir/f'fold_{fold:02d}_campaign_manifest.json'
    chunk_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    chunks=sorted((item for item in chunk_manifest['chunks'] if item.get('status')=='completed'),key=lambda x:int(x['chunk']))
    issues=[]
    if progress.get('status')!='paused_checkpointed':issues.append('progress_not_paused_checkpointed')
    if progress.get('frozen_test_accessed') is not False:issues.append('frozen_test_flag_not_false')
    if int(progress.get('total_chunks_committed',-1))!=len(chunks):issues.append('global_chunk_count_mismatch')
    if [int(x['chunk']) for x in chunks]!=list(range(1,len(chunks)+1)):issues.append('chunk_sequence_gap')
    if not chunks:issues.append('no_completed_chunks')
    fingerprint=chunk_manifest['fingerprint']
    latest_config=None; last_state=None; resources=[]; previous_batch=0; previous_records=0
    for item in chunks:
        cp=chunk_dir/item['checkpoint']
        if not cp.is_file():issues.append(f"missing_checkpoint:{item['chunk']}");continue
        body=json.loads(cp.read_text(encoding='utf-8'))
        latest_config=body.get('configuration')
        state=checkpointing.load_qnn_chunk_checkpoint(cp,fold,int(item['chunk']),fingerprint,latest_config)['state']
        if checkpointing.file_sha256(cp)!=item.get('sha256'):issues.append(f"checkpoint_hash_mismatch:{item['chunk']}")
        receipt=state.get('processed_range') or {}
        if int(receipt.get('first_batch',-1))!=previous_batch:issues.append(f"batch_range_gap:{item['chunk']}")
        if int(receipt.get('next_batch',-1))-int(receipt.get('first_batch',-1))!=16:issues.append(f"batch_range_size:{item['chunk']}")
        if int(state.get('processed_records',-1))<=previous_records:issues.append(f"processed_record_counter_not_increasing:{item['chunk']}")
        if int(state.get('processed_records',-1))-previous_records!=int(receipt.get('records',-1)):issues.append(f"record_counter_delta_mismatch:{item['chunk']}")
        if not state.get('gradient_check_result',{}).get('passed'):issues.append(f"gradient_check_failed:{item['chunk']}")
        if state.get('parameter_update_verified') is not True:issues.append(f"parameter_update_not_verified:{item['chunk']}")
        previous_batch=int(receipt.get('next_batch',0));previous_records=int(state.get('processed_records',0))
        resources.append({'available_ram_mib':int(state.get('available_ram_bytes',0))/2**20,
                          'peak_rss_mib':int(state.get('peak_rss_bytes',0))/2**20})
        last_state=state
    if last_state:
        for key,value in [('next_batch',last_state['next_batch']),('next_epoch',last_state['epoch']),('active_chunk',last_state['chunk_index'])]:
            if int(progress.get(key,-1))!=int(value if key!='active_chunk' else value):
                # Progress stores active_chunk as the committed chunk index.
                issues.append(f"progress_{key}_does_not_match_checkpoint")
    cfg=progress['configuration']; fold_rows=progress['stream_audit']['fold_record_counts']
    expected_updates_per_epoch=sum(math.ceil((checkpointing.DEVELOPMENT_ROWS-int(count))/int(cfg['batch_size']))
                                   for count in fold_rows.values())
    expected_updates=expected_updates_per_epoch*int(cfg['epochs'])
    observed_seconds_per_update=(float(last_state['training_seconds'])/int(progress['total_optimizer_updates'])
                                 if last_state and int(progress['total_optimizer_updates']) else None)
    report={'schema':'qmutbench-full-development-progress-audit-v1','status':'VERIFIED' if not issues else 'FAILED',
        'issues':issues,'campaign_status':progress.get('status'),'mode':progress.get('configuration',{}).get('mode','full-development'),
        'active_fold':fold,'committed_chunks':len(chunks),'latest_checkpoint':chunks[-1]['checkpoint'] if chunks else None,
        'next_epoch':last_state.get('epoch') if last_state else None,'next_batch':last_state.get('next_batch') if last_state else None,
        'processed_training_records':previous_records,'optimizer_updates':int(progress.get('total_optimizer_updates',0)),
        'cumulative_training_seconds':float(last_state.get('training_seconds',0)) if last_state else None,
        'cumulative_circuit_executions':int(last_state.get('training_executions',0)) if last_state else None,
        'estimated_total_optimizer_updates':expected_updates,
        'observed_training_seconds_per_optimizer_update':observed_seconds_per_update,
        'estimated_total_qnn_training_hours_at_observed_rate':(observed_seconds_per_update*expected_updates/3600
            if observed_seconds_per_update is not None else None),
        'estimate_note':'Extrapolation from the current fold-1 chunked QNN update rate; excludes streaming/preprocessing and classical evaluation; not an observed completion time.',
        'all_committed_chunks_have_gradient_and_parameter_update_checks':not any('gradient_check_failed' in x or 'parameter_update_not_verified' in x for x in issues),
        'minimum_available_ram_mib':min((r['available_ram_mib'] for r in resources),default=None),
        'maximum_checkpointed_peak_rss_mib':max((r['peak_rss_mib'] for r in resources),default=None),
        'source_audit':progress.get('stream_audit'),'frozen_test_accessed':False,
        'preflight_configuration':progress.get('configuration')}
    (output/'progress_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if issues:raise AssertionError(f'Full-development progress audit failed: {issues[:8]}')

if __name__=='__main__':main()
