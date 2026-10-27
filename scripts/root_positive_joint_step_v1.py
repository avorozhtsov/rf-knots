"""F-primary joint gradients with root-only MCTS single-success supervision."""
import math
from pathlib import Path
import torch
import train_joint_f_r2_r4_v1 as base
from run_continuous_train_mcts_v1 import read
from rf_knots import mcts_loop as M,reference
from rf_knots.evidence import SemanticAction

BASE_STEP=None
GROUPS=None
CAP=.05


class DeferredOptimizer:
    def __init__(self,optimizer):self.optimizer=optimizer;self.calls=0
    def __getattr__(self,name):return getattr(self.optimizer,name)
    def step(self,*args,**kwargs):
        self.calls+=1
        if self.calls!=1:raise ValueError('anchor attempted multiple AdamW steps')


def capped_scale(f_norm,positive_norm,cap=CAP):
    if not math.isfinite(f_norm) or f_norm<=1e-12 or not math.isfinite(positive_norm) or positive_norm<0:
        raise ValueError('invalid F/positive gradient norm')
    return min(1.,cap*f_norm/positive_norm) if positive_norm else 0.


def initialize(bank):
    global GROUPS
    grouped={}
    m=read(bank/'manifest.json')
    for e in m['paths']:
        p=bank/e['file']
        if base.sha(p)!=e['sha256']:raise ValueError('positive path pin differs')
        row=read(p);grouped.setdefault(tuple(row['query_group']),[]).append(row)
    if not grouped:raise ValueError('empty positive TRAIN bank')
    GROUPS=[(key,grouped[key]) for key in sorted(grouped)]


def select(step):
    ordinal=step-351232
    if ordinal<0 or GROUPS is None:raise ValueError('positive sampler not initialized')
    key,rows=GROUPS[ordinal%len(GROUPS)];weight=(10,1000)[(ordinal//len(GROUPS))%2]
    row=min(rows,key=lambda r:(weight*r['achieved']['cc']+r['achieved']['moves'],r['path_id']))
    data={**row['instance'],'cc_weight':weight};instance=M.Instance(**{**data,'word':tuple(data['word'])})
    return row,instance


def joint_step(net,objective,head,optimizer,group,step):
    if BASE_STEP is None:raise ValueError('anchor step absent')
    deferred=DeferredOptimizer(optimizer)
    metrics=BASE_STEP(net,objective,head,deferred,group,step)
    if deferred.calls!=1:raise ValueError('anchor did not defer exactly one step')
    parameters=[*net.parameters(),*objective.parameters(),*head.parameters()]
    old=base.gradient_snapshot(parameters)
    norm=float(metrics['combined_grad_norm']);old_clip=min(1.,1./(norm+1e-6))
    f=metrics['F_encoder_grad_norm']*old_clip
    row,instance=select(step)
    if row['gradient_scope']!='exact_zero_spent_TRAIN_root_only' or row['origin']!='MCTS_best_success_not_reference':
        raise ValueError('new positive scope/origin differs')
    state=M.SearchState(instance.word,instance.strands,0,0)
    action=SemanticAction(**row['root_action']).to_flat(instance.spec)
    if not reference.is_legal(instance.spec,instance.word,instance.strands,action,True):raise ValueError('illegal targetaction')
    actions=tuple(sorted(set(base.proposal_candidates(instance,state,4))|{action}))
    optimizer.zero_grad(set_to_none=True)
    logp=base.policy_log_probabilities(net,[(instance,state,actions)])
    policy=-logp[0,actions.index(action)]
    value=base.auxiliary_loss(net,head,instance,row['achieved'])
    loss=policy+.1*value
    if not torch.isfinite(loss) or float(loss.detach())<0:raise ValueError('nonfinite positive loss')
    loss.backward();positive=base.gradient_snapshot(parameters)
    positive_norm=base.encoder_norm(positive,parameters,net.encoder);scale=capped_scale(f,positive_norm)
    effective_old_aux=metrics['aux_encoder_grad_norm']*metrics['aux_scale']*old_clip
    effective_rehearsal=metrics['rehearsal_encoder_grad_norm']*metrics['rehearsal_scale']*old_clip
    aux_bound=effective_old_aux+positive_norm*scale
    if aux_bound>.100001*f or effective_rehearsal>.200001*f:raise ValueError('joint effective ratio cap exceeded')
    optimizer.zero_grad(set_to_none=True)
    for p,g,pos in zip(parameters,old,positive,strict=True):
        terms=([g] if g is not None else [])+([pos*scale] if pos is not None and scale else [])
        if terms:p.grad=sum(terms[1:],terms[0])
    combined=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    if not math.isfinite(float(combined)):raise ValueError('nonfinite combined gradient')
    optimizer.step()
    return {**metrics,'F_encoder_raw_grad_norm':metrics['F_encoder_grad_norm'],'F_encoder_grad_norm':f,
        'aux_encoder_grad_norm':aux_bound,'aux_scale':1.,'rehearsal_encoder_grad_norm':effective_rehearsal,
        'rehearsal_scale':1.,'combined_grad_norm':float(combined),'positive_loss':float(loss.detach()),
        'positive_policy_loss':float(policy.detach()),'positive_value_loss':float(value.detach()),
        'positive_encoder_grad_norm':positive_norm,'positive_scale':scale,'positive_effective_F_ratio':positive_norm*scale/f,
        'positive_path_id':row['path_id'],'positive_query_group':row['query_group'],'positive_cc_weight':instance.cc_weight,
        'positive_scope':'exact_zero_spent_TRAIN_root_only','anchor_clip_factor':old_clip,'real_AdamW_steps_this_update':1}
