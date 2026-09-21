"""Freeze several replay-verified training sizes with unchanged knot ownership."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from rf_knots.embedding_replay_corpus import lift, verify_record


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.source.read_text())
    source_hash = sha(args.source)
    result = copy.deepcopy(data)
    dimensions = []
    for i, row in enumerate(result['train']):
        witness = verify_record(row)
        if i % 3:
            length, strands = (60, 9) if i % 3 == 1 else (128, 17)
            expanded, prefix = lift(witness, strands, length)
            row['witness'] = expanded.to_dict()
            row['pair_prefix_length'] += prefix
            row['interval']['left'] = expanded.start.to_dict()
            row['interval']['witness_semantic_length'] += prefix
            row['training_size_view'] = {'length_floor': length, 'strands_floor': strands}
        checked = verify_record(row)
        dimensions.append((len(checked.start.word), checked.start.strands))
        if (i + 1) % 256 == 0:
            print(json.dumps({'verified_train': i + 1}), flush=True)
    for split in ('validation', 'size_shift'):
        for row in result[split]:
            verify_record(row)
    if sha(args.source) != source_hash:
        raise ValueError('source changed during construction')
    result['multisize_source_sha256'] = source_hash
    result['multisize_contract'] = (
        'Every third train root unchanged; other roots lifted by verified zero-CC moves '
        'to length>=60/N>=9 or length>=128/N>=17. Same knot types, not new independent knots. '
        'Validation views unchanged and are not necessarily larger than this training set.'
    )
    output = args.out / 'corpus.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    manifest = {
        'status': 'completed', 'source': str(args.source), 'source_sha256': source_hash,
        'corpus_sha256': sha(output), 'train_records': len(dimensions),
        'length_range': [min(x[0] for x in dimensions), max(x[0] for x in dimensions)],
        'strand_range': [min(x[1] for x in dimensions), max(x[1] for x in dimensions)],
        'code_sha256': {str(p): sha(p) for p in [Path(__file__), Path('src/rf_knots/embedding_replay_corpus.py')]},
        'contract': result['multisize_contract'],
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
