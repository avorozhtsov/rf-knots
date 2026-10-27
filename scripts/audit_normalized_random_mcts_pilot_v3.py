"""Independent canonical replay relative-path audit, new view namespace."""
from pathlib import Path
import audit_random_representation_mcts_v1 as auditor
auditor.DATA=Path('artifacts/normalized-random-mcts-pilot-v3')
if __name__=='__main__':auditor.main()
