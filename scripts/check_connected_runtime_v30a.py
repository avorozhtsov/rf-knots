"""Strict post-resume acceptance: saved metadata alone is not live progress."""
import argparse
from pathlib import Path
from check_connected_learning_live_v30 import check
from run_continuous_train_mcts_v1 import read

def verify_started(live,durable,status,after_step):
    assert status=='VERIFIED_ACTIVE','awaiting verified continuous progress'
    assert live>max(durable,after_step),'saved checkpoint metadata is not resumed training'

def main():
    p=argparse.ArgumentParser();p.add_argument('--after-step',type=int,required=True);a=p.parse_args()
    result=check();state=read(Path('artifacts/connected-learning-live-v30/state.json'))
    verify_started(result['LIVE'],result['durable'],state['status'],a.after_step)
    result.pop('v30_random_path_ids',None);print(result)

if __name__=='__main__':main()
