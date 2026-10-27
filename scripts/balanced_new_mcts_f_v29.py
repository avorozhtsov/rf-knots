"""Two gradient budgets only. No loss-magnitude proxy and no extra auxiliaries."""
import math
from pathlib import Path
import torch
import train_joint_f_r2_r4_v1 as base
import train_joint_f_r2_r4_bounded_v2 as bounded
import positive_sampler_v28 as sampler
from rf_knots import mcts_loop as M,reference
from rf_knots.evidence import SemanticAction

ORIGIN=None
EMA_F=None
COUNTS=None
SETTINGS=None


def mcts_weight(step,origin):
    n=step-origin
    if n<0:raise ValueError('balance clock precedes immutable origin')
    return .2 if n<2000 else .35 if n<4000 else .5


def full_norm(grads):
    return math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads if g is not None))


def scales(f,m,weight,ema):
    if not all(math.isfinite(x) and x>1e-12 for x in [f,m,ema]):
        raise ValueError('zero/nonfinite balanced gradient')
    if not 0<weight<1:raise ValueError('invalid gradient budget')
    target=min(ema,f,10*m)
    return (1-weight)*target/f,weight*target/m,target


def coefficients(group):
    if group.get('gradient_scope')!='exact_zero_spent_root_only':
        inst,_,c=base.group_data(group);return inst,c
    d=group['instance'];inst=M.Instance(**{**d,'word':tuple(d['word'])})
    state=M.SearchState(inst.word,inst.strands,0,0);c={}
    for item in group['root_coefficients']:
        a=item['action'];v=float(item['value'])
        if not math.isfinite(v) or not reference.is_legal(inst.spec,inst.word,inst.strands,a,True):
            raise ValueError('invalid root F term')
        c[(state,a)]=v
    if len(c)!=group['signal_terms']:raise ValueError('F term count differs')
    return inst,c


def initialize(s,saved=None):
    global ORIGIN,EMA_F,COUNTS,SETTINGS
    SETTINGS=s;ORIGIN=s['balance_origin_step']
    metrics=(saved or {}).get('metrics',{})
    EMA_F=metrics.get('balance_ema_f')
    COUNTS=dict(retained=0,random=0,new_retained=0,new_random=0,distinct_new_path_ids=[])
    if saved and saved['step']>s['parent_step']:
        COUNTS=metrics['connected_consumption']
    for source,prefix in [('retained','positive'),('random','random_positive')]:
        allowed=set(s['fresh_'+source+'_ids']);groups=[];found=set()
        for group,views in sampler.load_bank(Path(s[prefix+'_bank'])):
            kept=[]
            for view,rows in views:
                selected=[r for r in rows if r['path_id'] in allowed]
                found.update(r['path_id'] for r in selected)
                if selected:kept.append((view,selected))
            if kept:groups.append((group,kept))
        if not groups or found!=allowed:raise ValueError('empty or unpinned fresh MCTS source')
        sampler.SOURCES[source]=groups


def train_step(net,objective,head,optimizer,group,step):
    global EMA_F
    parameters=[*net.parameters(),*objective.parameters(),*head.parameters()]
    inst,c=coefficients(group)
    optimizer.zero_grad(set_to_none=True)
    f_loss,pairs=bounded.bounded_f_loss(net,inst,c);f_loss.backward()
    fg=base.gradient_snapshot(parameters);fn=full_norm(fg)
    optimizer.zero_grad(set_to_none=True)
    row,inst=sampler.select(step);source=sampler.LAST['source'];ident=row['path_id']
    assert ident in SETTINGS['fresh_'+source+'_ids']
    assert row['origin']=='MCTS_best_success_not_reference' and row['gradient_scope']=='exact_zero_spent_TRAIN_root_only'
    state=M.SearchState(inst.word,inst.strands,0,0)
    action=SemanticAction(**row['root_action']).to_flat(inst.spec)
    assert reference.is_legal(inst.spec,inst.word,inst.strands,action,True)
    actions=tuple(sorted(set(base.proposal_candidates(inst,state,4))|{action}))
    policy=-base.policy_log_probabilities(net,[(inst,state,actions)])[0,actions.index(action)]
    value=base.auxiliary_loss(net,head,inst,row['achieved'])
    mloss=policy+.1*value
    if not torch.isfinite(mloss) or float(mloss.detach())<0:raise ValueError('invalid MCTS loss')
    mloss.backward();mg=base.gradient_snapshot(parameters);mn=full_norm(mg)
    candidate_ema=fn if EMA_F is None else .99*EMA_F+.01*fn
    w=mcts_weight(step,ORIGIN);fs,ms,target=scales(fn,mn,w,candidate_ema)
    applied_f=full_norm([g*fs if g is not None else None for g in fg])
    applied_m=full_norm([g*ms if g is not None else None for g in mg])
    cosine=sum(float((a.float()*b.float()).sum().cpu()) for a,b in zip(fg,mg) if a is not None and b is not None)/(fn*mn)
    optimizer.zero_grad(set_to_none=True)
    for p,a,b in zip(parameters,fg,mg,strict=True):
        terms=([a*fs] if a is not None else [])+([b*ms] if b is not None else [])
        if terms:p.grad=sum(terms[1:],terms[0])
    combined=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step();EMA_F=candidate_ema
    COUNTS[source]+=1;COUNTS['new_'+source]+=1
    if ident not in COUNTS['distinct_new_path_ids']:COUNTS['distinct_new_path_ids'].append(ident)
    clip=min(1.,1./(float(combined)+1e-6))
    ef=base.encoder_norm(fg,parameters,net.encoder);em=base.encoder_norm(mg,parameters,net.encoder)
    return dict(F_loss=float(f_loss.detach()),positive_loss=float(mloss.detach()),
        positive_policy_loss=float(policy.detach()),positive_value_loss=float(value.detach()),
        F_encoder_grad_norm=ef*fs,mcts_encoder_grad_norm=em*ms,
        aux_loss=0.,aux_scale=0.,aux_encoder_grad_norm=0.,
        rehearsal_scale=0.,rehearsal_encoder_grad_norm=0.,imitation_loss=0.,distance_loss=0.,
        combined_grad_norm=float(combined),sampled_F_terms=len(c),F_pairs=pairs,
        balance_ema_f=EMA_F,balance_origin_step=ORIGIN,balance_update=step-ORIGIN+1,
        F_raw_full_norm=fn,MCTS_raw_full_norm=mn,F_gradient_scale=fs,MCTS_gradient_scale=ms,
        F_effective_full_norm=applied_f*clip,MCTS_effective_full_norm=applied_m*clip,
        actual_MCTS_norm_share=applied_m/(applied_f+applied_m),target_MCTS_norm_share=w,
        F_MCTS_cosine=cosine,common_target_norm=target,final_clip_factor=clip,
        positive_sampling_source=source,positive_admitted_path_id=ident,
        positive_observation_shard=sampler.LAST['shard'],real_AdamW_steps_this_update=1,
        connected_consumption={**COUNTS,'distinct_new_path_ids':list(COUNTS['distinct_new_path_ids'])},
        consumed_positive_bank_sha256=SETTINGS['positive_bank_sha256'],
        consumed_random_bank_sha256=SETTINGS['random_positive_bank_sha256'])


def numerical_gate(metrics):
    for k in ['F_loss','positive_loss','F_raw_full_norm','MCTS_raw_full_norm',
              'F_effective_full_norm','MCTS_effective_full_norm','combined_grad_norm','balance_ema_f']:
        if not math.isfinite(metrics[k]) or metrics[k]<0:raise ValueError('invalid balanced state:'+k)
    if min(metrics['F_raw_full_norm'],metrics['MCTS_raw_full_norm'])<=1e-12:raise ValueError('vanishing balanced component')
    if abs(metrics['actual_MCTS_norm_share']-metrics['target_MCTS_norm_share'])>1e-6:raise ValueError('gradient budget mismatch')
    if metrics['aux_loss']!=0 or metrics['rehearsal_encoder_grad_norm']!=0:raise ValueError('auxiliary outside budget')
    assert metrics['real_AdamW_steps_this_update']==1
