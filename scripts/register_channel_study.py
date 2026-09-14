"""Freeze matched-media and segmentation-grounded object-swap diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cta.verification_workbench import load_packet,select_items,sha,canonical,now,write_json
from scripts.channel_study_lib import (MEDIA_ARMS,BINDING_ARMS,assessment,options,
    supplied,neutral_fields,record_pixels,card_image,code_hashes)

CLASSES=['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
         'chair','cow','dining table','dog','horse','motorbike','person','potted plant','sheep','sofa','train','television']


def decode_mask(maskraw):
    mask=np.asarray(Image.open(io.BytesIO(maskraw)))
    if mask.ndim==2:return mask
    if mask.ndim!=3 or mask.shape[2]!=3:raise ValueError('unsupported VOC mask encoding')
    # The archive stores RGB visualizations of the official VOC indexed palette.
    palette=[]
    for label in range(256):
        value=label;rgb=[0,0,0]
        for bit in range(8):
            for channel in range(3):rgb[channel]|=((value>>channel)&1)<<(7-bit)
            value>>=3
        palette.append((rgb[0]<<16)|(rgb[1]<<8)|rgb[2])
    order=np.argsort(palette);keys=np.array(palette)[order]
    packed=(mask[:,:,0].astype(np.int32)<<16)|(mask[:,:,1].astype(np.int32)<<8)|mask[:,:,2]
    indices=np.searchsorted(keys,packed)
    if np.any(indices>=256) or not np.array_equal(keys[np.minimum(indices,255)],packed):
        raise ValueError('RGB mask contains a color outside the VOC palette')
    return order[indices].astype(np.uint8)


def object_pool(parquet):
    import pyarrow.parquet as pq
    buckets=defaultdict(list)
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=16):
        for index,row in enumerate(batch.to_pylist()):
            raw=row['image']['bytes'];maskraw=row['mask']['bytes']
            mask=decode_mask(maskraw)
            ih=hashlib.sha256(raw).hexdigest()
            for label in sorted(set(np.unique(mask))-{0,255}):
                if not 1<=label<=20:raise ValueError('unexpected VOC category')
                keep=mask==label;yy,xx=np.where(keep)
                if keep.sum()<1000 or keep.mean()<.025 or np.ptp(xx)<40 or np.ptp(yy)<40:continue
                buckets[int(label)].append({'raw':raw,'maskraw':maskraw,'class_id':int(label),
                    'label':CLASSES[label],'source_sha256':ih,'mask_sha256':hashlib.sha256(maskraw).hexdigest()})
    for values in buckets.values():values.sort(key=lambda v:v['source_sha256'])
    ordered=[bucket[j] for j in range(max(map(len,buckets.values())))
             for _,bucket in sorted(buckets.items()) if j<len(bucket)]
    used=set();selected=[]
    for obj in ordered:
        if obj['source_sha256'] in used:continue
        selected.append(obj);used.add(obj['source_sha256'])
    return selected


def cutout(obj):
    rgb=np.asarray(Image.open(io.BytesIO(obj['raw'])).convert('RGB'))
    mask=decode_mask(obj['maskraw'])==obj['class_id']
    if rgb.shape[:2]!=mask.shape:raise ValueError('source/mask dimensions differ')
    yy,xx=np.where(mask);out=np.full_like(rgb,255);out[mask]=rgb[mask]
    return Image.fromarray(out[yy.min():yy.max()+1,xx.min():xx.max()+1])


def build(a):
    rows=[]
    for n in [0,1]:rows.extend(load_packet(a.confirmation/f'packet_{n}')[1])
    ids=sorted({r['item_id'] for r in rows})
    if len(ids)!=128:raise ValueError('expected exactly the archived confirmation population')
    lookup={(r['item_id'],r['condition']):r for r in rows}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'assets').mkdir()
    cases=[];requests=[[],[]]
    def image_asset(image):
        b=io.BytesIO();image.save(b,format='PNG');data=b.getvalue()
        name='assets/'+hashlib.sha256(data).hexdigest()+'.png'
        path=a.output/name
        if not path.exists():path.write_bytes(data)
        return name
    def request(shard,study,item,state,arm,prompt,image,gold,mapping,fields=None,tokens=96):
        row={'key':canonical([study,item,state,arm]),'study':study,'item_id':item,
            'state':state,'arm':arm,'prompt':prompt,'image':image,
            'image_sha256':sha(a.output/image) if image else None,'tokens':tokens,
            'gold':gold,'option_map':mapping,'read_target':fields}
        requests[shard].append(row)
    for n,item in enumerate(ids):
        base=lookup[item,'record_true'];mapping=base['option_map'];assumption=base['record']['assumption']
        for condition in ['record_true','record_false']:
            row=lookup[item,condition];fields=row['registered_read_text']
            scene=Image.open(row['image_path']).convert('RGB')
            scene_name=image_asset(scene)
            cropped=record_pixels(scene,Image.open(row['mask_path']))
            record_name=image_asset(cropped)
            gold='consistent' if condition=='record_true' else 'inconsistent'
            prompt=assessment(assumption,mapping)
            for arm in MEDIA_ARMS:
                image=scene_name if arm in ['scene','oracle_scene'] else record_name
                if arm=='oracle_text':image=None
                request(n%2,'media',item,condition,arm,
                    prompt+(supplied(fields) if arm.startswith('oracle') else ''),image,gold,mapping)
            request(n%2,'media',item,condition,'read',
                'Transcribe the measurement and uncertainty fields verbatim, including the object-name field. Ignore the title and RECORD ID. Preserve all values and units; do not correct the record.',
                scene_name,None,mapping,fields,384)
            cases.append({'study':'media','item_id':item,'state':condition,'family':row['family'],
                'source':'coco' if item.startswith('coco') else 'voc','image':scene_name,
                'record_image':record_name,'fields':fields,'assumption':assumption,'gold':gold,
                'source_sha256':row['source_sha256'],'original_image_sha256':row['image_sha256'],
                'carrier_mask_sha256':row['mask_sha256']})
    templates=select_items(rows,64)
    tids=sorted({r['item_id'] for r in templates})
    pool=object_pool(a.voc_parquet);used=set();pair_meta=[]
    for n,template in enumerate(tids):
        target=next(o for o in pool if o['source_sha256'] not in used);used.add(target['source_sha256'])
        foil=next(o for o in pool if o['source_sha256'] not in used and o['class_id']!=target['class_id']);used.add(foil['source_sha256'])
        target_photo,foil_photo=cutout(target),cutout(foil)
        item=f'binding-{n:03d}';base=lookup[template,'record_true'];assumption=base['record']['assumption'];mapping=base['option_map']
        valid=neutral_fields(lookup[template,'record_true']);invalid=neutral_fields(lookup[template,'record_false'])
        left_valid=n%2==0;left_fields,right_fields=(valid,invalid) if left_valid else (invalid,valid)
        pictures=[]
        for state,left_target in [('target_left',True),('target_right',False)]:
            pic=card_image(target_photo if left_target else foil_photo,foil_photo if left_target else target_photo,
                           left_fields,right_fields,a.font)
            pictures.append(np.asarray(pic));image=image_asset(pic)
            gold='consistent' if left_target==left_valid else 'inconsistent'
            target_fields=left_fields if left_target else right_fields
            prompt=f'Find the panel showing the object category "{target["label"]}". Assess only the record directly beneath that object. Assumptions: '+assumption+' '+options(mapping)
            fields_cue=supplied({'LEFT':left_fields,'RIGHT':right_fields})
            location_cue=f'\nA supplied object-location annotation places the requested {target["label"]} in the '+('LEFT' if left_target else 'RIGHT')+' panel.'
            for arm in BINDING_ARMS:
                request(n%2,'binding',item,state,arm,
                    prompt+(fields_cue if arm in ['fields','both'] else '')+(location_cue if arm in ['location','both'] else ''),image,gold,mapping)
            request(n%2,'binding',item,state,'localize',
                f'Which panel shows the object category "{target["label"]}"? Answer L for LEFT or R for RIGHT, with that letter only.',
                image,'L' if left_target else 'R',None)
            request(n%2,'binding',item,state,'text_reasoning',assessment(assumption,mapping)+supplied(target_fields),None,gold,mapping)
            cases.append({'study':'binding','item_id':item,'state':state,'family':base['family'],
                'source':'voc_composite','image':image,'fields':{'LEFT':left_fields,'RIGHT':right_fields},
                'assumption':assumption,'gold':gold,'target_label':target['label'],
                'foil_label':foil['label'],'target_location':'LEFT' if left_target else 'RIGHT',
                'template_item':template})
        if not np.array_equal(pictures[0][370:],pictures[1][370:]):raise ValueError('record pixels changed on object swap')
        pair_meta.append({'item_id':item,'target':{k:v for k,v in target.items() if k not in ['raw','maskraw']},
                         'foil':{k:v for k,v in foil.items() if k not in ['raw','maskraw']}})
    write_json(a.output/'cases.json',cases)
    hashes={}
    for shard,req in enumerate(requests):
        path=a.output/f'requests_{shard}.jsonl';path.write_text(''.join(canonical(r)+'\n' for r in req),encoding='utf-8')
        if len(req)!=1152 or len({r['key'] for r in req})!=1152:raise ValueError('unexpected call coverage')
        hashes[str(shard)]=sha(path)
    models={}
    for name,path in [('qwen7',a.qwen7),('qwen3',a.qwen3)]:
        cfg=yaml.safe_load(path.read_text())['model']
        if cfg.get('do_sample'):raise ValueError('diagnostic requires deterministic decoding')
        models[name]={'config':cfg,'checkpoint_config_sha256':sha(Path(cfg['name_or_path'])/'config.json')}
    for name in code_hashes():
        dst=a.output/'executed_source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(__file__).resolve().parents[1]/name,dst)
    reg={'frozen_at':now(),'media_items':128,'binding_items':64,'expected_calls_per_model':2304,
        'expected_total_calls':4608,'expected_calls_per_shard':1152,'models':models,'code':code_hashes(),
        'request_hashes':hashes,'cases_sha256':sha(a.output/'cases.json'),
        'assets':{p.relative_to(a.output).as_posix():sha(p) for p in (a.output/'assets').iterdir()},
        'object_pairs':pair_meta,'font_sha256':sha(a.font),'voc_parquet_sha256':sha(a.voc_parquet),
        'selection':'Reuse all 128 archived mitigation-confirmation scenes; no outcome inputs. Binding templates select 64 by existing family-balanced ID-hash selection. VOC objects use class-round-robin source-hash order, distinct source photos and different classes per pair; at least 1000 annotated pixels, 2.5% image area, bbox extents >=40px.',
        'scope':'Prospectively frozen diagnostic on reused records and controlled digital object composites. Not unseen-source confirmation, natural photographs of records, certified physical truth, deployable oracle defense, or neural-mechanism identification.',
        'media_primary':[{'a':'scene','b':'oracle_scene'},{'a':'oracle_scene','b':'oracle_text'},{'a':'scene','b':'record_only'}],
        'binding_primary':[{'a':'fields','b':'both'},{'a':'location','b':'both'}],
        'primary_endpoint':'All-item paired accuracy: both valid/false records for media; both object-swap states for binding.',
        'tests':'Two-sided exact paired McNemar. Separate prespecified Holm families: six media contrasts (three per model), four binding contrasts (two per model).',
        'ci':'10000 paired bootstrap draws, seed 20260914; stratify by COCO/VOC for media, by eight template families for binding.',
        'secondary':'Individual-state accuracies, all other contrasts, independent localization/text-reasoning controls, and binding failures among items passing both controls are descriptive. No post-result pooling or conditional superiority claims.',
        'budget':'96 output tokens per decision/localization, 384 per independent transcription. All decisions one call, greedy decoding. Supplied annotations are oracle diagnostics, not generated model outputs.',
        'stopping':'Every frozen cell once. Retain errors/unparsed responses, no outcome-dependent item/prompt changes, no significance-based stopping, no silent retries.'}
    write_json(a.output/'registration.json',reg)
    print(json.dumps({'media_items':128,'binding_items':64,'calls':4608,'assets':len(reg['assets']),'registration_sha256':sha(a.output/'registration.json')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['confirmation','voc-parquet','font','qwen7','qwen3','output']:p.add_argument('--'+n,type=Path,required=True)
    build(p.parse_args())
