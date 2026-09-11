"""Read-only classification of four explicit synthetic-training BC batches."""
import collections
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from q4_rl.micro_network import configure_cpu, load_policy, pack_observations

configure_cpu()
started = time.perf_counter()
root = Path('../q4-rl-micro-actions/results/q4_rl/server-pair-v2-complete-001/micro512/training')
checkpoint = root/'warmstart.pt'
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert digest == 'cb5df8dd338cc740222fc0edd5259ea63a64ddd7cb12398d245e781eb7843757'
model = load_policy(checkpoint, deterministic=True).model
groups = collections.defaultdict(lambda: collections.Counter())
inputs, losses, complete_seeds, teacher_progress = [], [], [], []
for batch_index in range(4):
    path = root/f'batch-{batch_index:06d}-attempt-{batch_index:06d}.json.gz'
    compressed = path.read_bytes()
    inputs.append({'file':path.name, 'sha256':hashlib.sha256(compressed).hexdigest()})
    payload = json.loads(gzip.decompress(compressed))
    del compressed
    if isinstance(payload, list): episodes = payload
    else:
        episodes = payload.get('episodes', payload.get('rows', payload.get('batch')))
        if not isinstance(episodes, list):
            raise ValueError(f'Unsupported raw schema: {list(payload)}')
    progress = json.loads((root/f'progress-{batch_index+1:06d}.json').read_text())
    teacher_progress.append(progress)
    losses.append({'batch':batch_index+1, **progress['update']})
    assert len(episodes) == 16
    records, tags = [], []
    for episode in episodes:
        assert not episode.get('administrative_skip')
        complete_seeds.append(episode['seed'])
        steps = episode['controller_learning']['micro_steps']
        assert len(steps) == len(episode['records'])
        for record, step in zip(episode['records'], steps):
            assert record['action_kind'] == step['kind']
            records.append(record)
            tags.append((step['kind'], step['role']))
    with torch.no_grad():
        for offset in range(0, len(records), 128):
            chunk = records[offset:offset+128]
            global_features, features, mask = pack_observations(chunk)
            logits, _ = model(global_features, features, mask)
            actions = torch.tensor([row['action_index'] for row in chunk])
            indices = torch.arange(len(chunk))
            predictions = logits.argmax(-1)
            log_probs = logits.log_softmax(-1)
            teacher_logp = log_probs[indices,actions]
            teacher_features = features[indices,actions]
            equivalent = (features == teacher_features[:,None,:]).all(-1) & mask
            multiplicity = equivalent.sum(-1)
            class_probability = (log_probs.exp()*equivalent).sum(-1)
            equiv_top1 = equivalent[indices,predictions]
            for j, (kind, role) in enumerate(tags[offset:offset+len(chunk)]):
                values = {'n':1, 'correct':int(predictions[j]==actions[j]),
                    'equivalent_top1':int(equiv_top1[j]), 'cross_entropy_sum':float(-teacher_logp[j]),
                    'teacher_probability_sum':float(teacher_logp[j].exp()),
                    'teacher_class_probability_sum':float(class_probability[j]),
                    'indistinguishable_teacher_count':int(multiplicity[j]>1),
                    'multiplicity_sum':int(multiplicity[j]),
                    'equal_features_ce_floor_sum':math.log(int(multiplicity[j]))}
                for key in ('all', f'kind:{kind}', f'role:{role}', f'kind_role:{kind}/{role}', f'batch:{batch_index+1}'):
                    groups[key].update(values)
    print(json.dumps({'batch':batch_index+1,'records':len(records),'elapsed_s':time.perf_counter()-started}),flush=True)
    del payload, episodes, records, tags, chunk, episode, steps, record, step
    gc.collect()
assert complete_seeds == list(range(8001000,8001064))
summaries = {}
for key, sums in groups.items():
    n=sums['n']
    summaries[key]={'records':n,'top1':sums['correct']/n,'float32_feature_equivalent_top1':sums['equivalent_top1']/n,
        'cross_entropy':sums['cross_entropy_sum']/n,'mean_teacher_probability':sums['teacher_probability_sum']/n,
        'mean_teacher_equivalence_class_probability':sums['teacher_class_probability_sum']/n,
        'fraction_teacher_has_indistinguishable_alternative':sums['indistinguishable_teacher_count']/n,
        'mean_teacher_class_size':sums['multiplicity_sum']/n,
        'mean_exact_feature_ce_floor':sums['equal_features_ce_floor_sum']/n,
        'ce_excess_over_exact_feature_floor':(sums['cross_entropy_sum']-sums['equal_features_ce_floor_sum'])/n}
result={'scope':'in-sample supervised classification on fixed synthetic-training teacher inputs; no counterfactual rollout or independent efficacy claim',
    'checkpoint_sha256':digest,'seed_range':[8001000,8001063],'episodes':64,'inputs':inputs,
    'groups':summaries,'online_BC_losses':losses,'elapsed_s':time.perf_counter()-started,
    'equivalence_definition':'exact equality of all 50 float32 candidate features within the same decision; same global/context for every candidate',
    'ce_floor_scope':'permutation-equivariant candidate MLP cannot assign unequal probabilities to exactly equal feature rows'}
teacher_t = sum(p['mean_actual_time_s']*p['batch_episodes'] for p in teacher_progress)
teacher_l = sum(p['mean_common_lower_bound_s']*p['batch_episodes'] for p in teacher_progress)
result['teacher_rollout_context'] = {'scope':'original teacher rollouts, not performance of the frozen model',
    'all_cleared':sum(p['full_clear'] for p in teacher_progress), 'episodes':64,
    'mean_actual_time_s':teacher_t/64, 'mean_common_lower_bound_s':teacher_l/64,
    'sum_time_over_sum_bound':teacher_t/teacher_l,
    'failed_clear_count':sum(p['failed_clear_count'] for p in teacher_progress)}
Path('handoff/bc-fit-diagnostic/results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
