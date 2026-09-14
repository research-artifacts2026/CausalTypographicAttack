"""Display full-population paired counts without selecting by model outcome."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import sha


def build(folder):
    d=json.loads((folder/'analysis.json').read_text())
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Input / independent control & Qwen2.5-VL-7B & Qwen3-VL-8B \\',r'\midrule',r'\multicolumn{3}{l}{\textit{Media: both record states correct}} \\']
    for study,arms in [('media',[('scene','Scene'),('record_only','Record pixels'),('oracle_scene','Scene + exact fields'),('oracle_text','Exact fields, no image')]),('binding',[('none','Image only'),('fields','Image + exact fields'),('location','Image + target location'),('both','Image + both annotations'),('localize','Independent localization'),('text_reasoning','Single-record text reasoning')])]:
        if study=='binding':lines.extend([r'\midrule',r'\multicolumn{3}{l}{\textit{Association: both object placements correct}} \\'])
        for arm,label in arms:
            values=[d['results'][m][study]['metrics'][arm] for m in ['qwen7','qwen3']]
            lines.append(' & '.join([label,*[f"{v['pair_correct']}/{v['pair_n']}" for v in values]])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    target=folder/'generated_main_table.tex';target.write_bytes(('\n'.join(lines)+'\n').encode())
    provenance={'analysis_sha256':sha(folder/'analysis.json'),'table_sha256':sha(target),'generator_sha256':sha(Path(__file__)),
                'scope':'Fixed display arms; complete five-arm media and four-arm association results and all ten tests remain in the appendix.'}
    (folder/'main_table_provenance.json').write_bytes((json.dumps(provenance,indent=2)+'\n').encode())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);build(p.parse_args().evidence)
