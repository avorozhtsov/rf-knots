"""Independent full episode replay using existing separate auditor."""
from pathlib import Path
import audit_random_representation_mcts_v1 as auditor
auditor.DATA=Path('artifacts/short-horizon-random-mcts-pilot-v4')
if __name__=='__main__':auditor.main()
