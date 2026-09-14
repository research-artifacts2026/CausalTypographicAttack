"""Generate the four registered comparison rows from completed analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def make(folder):
    analysis = folder / 'analysis.json'
    d = json.loads(analysis.read_text(encoding='utf-8'))
    if d['status'] != 'complete' or d['actual_calls'] != 1536 or len(d['primary_tests']) != 4:
        raise ValueError('completed four-contrast analysis required')
    names = {'qwen7': 'Qwen2.5-VL-7B', 'qwen3': 'Qwen3-VL-8B'}
    labels = {'direct': 'Direct', 'reasoned': 'Reason then answer'}
    lines = [r'\begin{tabular}{llrrr}', r'\toprule',
             r'Model & Read + rules minus & $\Delta$ (pp) & 95\% CI (pp) & $p_H$ \\', r'\midrule']
    for t in d['primary_tests']:
        p = r'$<.0001$' if t['holm_p'] < .0001 else f"{t['holm_p']:.4f}"
        lines.append(' & '.join([names[t['model']], labels[t['a']], f"{100*t['delta']:+.2f}",
                     f"[{100*t['ci95'][0]:.2f}, {100*t['ci95'][1]:.2f}]", p]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    out = folder / 'generated_tests.tex'
    out.write_bytes(('\n'.join(lines) + '\n').encode())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = {'analysis_sha256': sha(analysis), 'generator_sha256': sha(Path(__file__)),
                  'table_sha256': sha(out), 'p_display_threshold': .0001,
                  'scope': 'Four prospectively specified paired contrasts; Holm across four, marginal bootstrap intervals.'}
    (folder / 'test_table_provenance.json').write_bytes((json.dumps(provenance, indent=2) + '\n').encode())
    print('Generated four registered paired-comparison rows.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', type=Path, required=True)
    make(p.parse_args().evidence)
