"""Independent replay auditor on v2 immutable pilot namespace."""
from pathlib import Path
import audit_random_representation_mcts_v1 as audit

audit.DATA=Path('artifacts/random-representation-mcts-pilot-v2')
if __name__=='__main__':audit.main()
