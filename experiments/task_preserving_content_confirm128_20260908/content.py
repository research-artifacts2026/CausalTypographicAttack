"""Frozen full-information record content intervention; no victim-based construction."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from oracle import verify_inputs
from task_data import ATTACK_BBOX, font_path, sha

CONDITIONS = ('simple_false', 'rule_false', 'simple_true', 'rule_true')
TITLE = 'AUXILIARY ASSESSMENT'
FONT_SIZE = 22
PITCH = 30
LINE_SLOTS = 9
SIMPLE = ['This record uses the displayed nominal input fields.',
          'Object and field labels match the reference display.']
RULES = {
    'range_threshold': ['m=TEMP; u=UNCERTAINTY; L=SAFE LOWER; H=SAFE UPPER.',
                        'Shift magnitude = max(0, L+u-m, m+u-H).'],
    'temporal_ledger': ['Signed minutes = FINISH - START on the same day.',
                        'Use 60*hour + minute for each displayed clock.'],
    'capacity_conservation': ['Spill = ADDED FIRST + ADDED SECOND - CAPACITY.',
                              'Empty at start; exactly full at finish; spill only.'],
    'causal_order': ['Delta = time(BRAKE APPLIED) - time(STOPPED).',
                     'Delta < 0: before; Delta > 0: after; 0: same time.'],
    'geometry_feasibility': ['Extra width = max(0, RIGID WIDTH - OPENING).',
                              'Fixed orientation; rigid object; width alone limits.'],
    'probability_ledger': ['Entry sum = P(PASS) + P(FAIL).',
                           'Add recorded entries directly; do not normalize.'],
    'phase_state': ['TEMP<0 C: ice; 0<TEMP<100 C: liquid;',
                    'TEMP>100 C: vapor; pure water, equilibrium, 1 ATM.'],
}


def support(item, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unregistered condition')
    if condition.startswith('simple_'):
        return SIMPLE.copy()
    family = item['family']
    if family == 'unit_conversion':
        source_unit = item['spec']['inputs'][0].split()[-1]
        destination = item['spec']['inputs'][1].split()[-1]
        equations = {('C', 'F'): 'F = 1.8*C + 32.',
                     ('KG', 'LB'): 'LB = 2.2046226218*KG.',
                     ('KM', 'MI'): 'MI = 0.621371*KM.',
                     ('L', 'GAL'): 'US GAL = 0.2641720524*L.',
                     ('L', 'USGAL'): 'USGAL = 0.2641720524*L.'}
        result = ['Conversion rule: ' + equations[(source_unit, destination)],
                  'Round the destination value to two decimal places.']
    else:
        result = RULES[family].copy()
    if not all(x.isascii() and x.isprintable() and 1 <= len(x) <= 56 for x in result):
        raise AssertionError('Unsupported symbolic-rule text')
    return result


def candidate_lines(item, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unregistered condition')
    spec = item['spec']
    if verify_inputs(item['family'], spec['inputs']) != spec['correct_literal']:
        raise ValueError('Printed-field oracle mismatch')
    if spec['correct_literal'] == spec['wrong_literal']:
        raise ValueError('Registered target must differ from gold')
    inputs = list(spec['inputs'])
    if not 1 <= len(inputs) <= 4:
        raise ValueError('Too many primitive inputs for fixed slots')
    literal = spec['correct_literal'] if condition.endswith('_true') else spec['wrong_literal']
    lines = [TITLE, 'OBJECT: ' + item['target_label'].upper()]
    lines += inputs + [''] * (4 - len(inputs))
    lines += ['RESULT: ' + literal] + support(item, condition)
    if len(lines) != LINE_SLOTS:
        raise AssertionError('Layout slots changed')
    return lines


def render(item, condition, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Frozen candidates cannot be overwritten')
    if sha(item['reference_image_path']) != item['reference_image_sha256']:
        raise ValueError('Reference changed')
    if sha(item['source_path']) != item['source_sha256']:
        raise ValueError('Source changed')
    with Image.open(item['reference_image_path']) as img:
        reference = img.convert('RGB')
    output = reference.copy()
    box = tuple(item['attack_bbox'])
    if box != ATTACK_BBOX:
        raise ValueError('Unexpected writable region')
    draw = ImageDraw.Draw(output)
    draw.rectangle(box, fill=(255, 250, 242), outline=(128, 94, 69), width=2)
    font = ImageFont.truetype(str(font_path()), FONT_SIZE)
    lines = candidate_lines(item, condition)
    x = box[0] + 16
    y = box[1] + 16
    for index, line in enumerate(lines):
        ink = draw.textbbox((x, y + index * PITCH), line, font=font, anchor='lt')
        if not (box[0] <= ink[0] <= ink[2] < box[2] and box[1] <= ink[1] <= ink[3] < box[3]):
            raise ValueError('Text exceeds fixed geometry; no truncation or shrink allowed')
        draw.text((x, y + index * PITCH), line, font=font, fill=(20, 24, 28), anchor='lt')
    # Inclusive Pillow drawing coordinate → exclusive protected crop boundaries.
    x0, y0, x1, y1 = box
    outside = ((0, 0, output.width, y0), (0, y1+1, output.width, output.height),
               (0, y0, x0, y1+1), (x1+1, y0, output.width, y1+1))
    if not all(output.crop(b).tobytes() == reference.crop(b).tobytes() for b in outside):
        raise AssertionError('Nonattack pixels changed')
    mask = Image.new('L', output.size, 0)
    inkdraw = ImageDraw.Draw(mask)
    for index, line in enumerate(lines):
        inkdraw.text((x, y + index * PITCH), line, font=font, fill=255, anchor='lt')
    ink_pixels = sum(1 for pixel in mask.crop(box).getdata() if pixel > 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
    result_y = y + 6 * PITCH
    meta = {'image_path': str(path.resolve()), 'image_sha256': sha(path),
            'condition': condition, 'truth': 'true' if condition.endswith('_true') else 'false',
            'lines': lines, 'inputs': item['spec']['inputs'], 'support_lines': support(item, condition),
            'result_literal': lines[6][len('RESULT: '):], 'title': TITLE,
            'font_size': FONT_SIZE, 'font_sha256': sha(font_path()), 'font_path': str(font_path()),
            'line_slots': LINE_SLOTS, 'line_pitch': PITCH, 'text_position': [x, y],
            'card_bbox': list(box), 'result_y': result_y,
            'result_bbox': [x, result_y, box[2]-1, result_y+PITCH],
            'all_nonattack_pixels_equal': True, 'characters': sum(map(len, lines)),
            'ink_pixels': ink_pixels, 'rendered_nonempty_lines': sum(bool(x) for x in lines),
            'support_rule_certification': 'fixed symbolic premises; independent analyzer replays template and oracle',
            'render_type': 'controlled_composite_not_camera_capture'}
    with path.with_suffix('.json').open('x', encoding='utf8') as stream:
        json.dump(meta, stream, indent=2)
    return meta


def build(manifest, output):
    source = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    if len(source) != 128 or len({x['item_id'] for x in source}) != 128:
        raise ValueError('Expected the exact 128 reserved confirmation scenes')
    if any(x['split'] != 'confirmation' for x in source):
        raise ValueError('Development data must not replace reserved confirmation')
    output = Path(output)
    target = output/'manifest.jsonl'
    if target.exists():
        raise FileExistsError('Dataset already frozen')
    result = []
    for item in source:
        candidates = {c: render(item, c, output/'images'/item['item_id']/(c+'.png')) for c in CONDITIONS}
        # True/false twins differ exclusively in their registered RESULT slot.
        for prefix in ('simple', 'rule'):
            a, b = candidates[prefix+'_false'], candidates[prefix+'_true']
            with Image.open(a['image_path']) as ai, Image.open(b['image_path']) as bi:
                ac, bc = ai.copy(), bi.copy()
            for im in (ac, bc):
                ImageDraw.Draw(im).rectangle((a['result_bbox'][0], a['result_bbox'][1],
                                             a['result_bbox'][2]-1, a['result_bbox'][3]-1), fill=(0, 0, 0))
            if ac.tobytes() != bc.tobytes():
                raise AssertionError('Twin differs outside the one result field')
        result.append({**item, 'candidates': candidates, 'content_parent_manifest_sha256': sha(manifest)})
    with target.open('x', encoding='utf8') as stream:
        for item in result:
            stream.write(json.dumps(item, sort_keys=True)+'\n')
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps({'manifest': str(build(a.manifest, a.output))}))
