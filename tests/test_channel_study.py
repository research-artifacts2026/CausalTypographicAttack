import numpy as np
from PIL import Image
import pytest
from scripts.channel_study_lib import record_pixels,neutral_fields,assessment,card_image
from scripts.register_channel_study import decode_mask


def test_context_ablation_preserves_every_carrier_pixel():
    original=np.arange(300,dtype=np.uint8).reshape(10,10,3)
    mask=np.zeros((10,10),dtype=np.uint8);mask[2:7,3:8]=255
    result=np.asarray(record_pixels(Image.fromarray(original),Image.fromarray(mask)))
    assert np.array_equal(result[mask>0],original[mask>0])
    assert np.all(result[mask==0]==127)


def test_voc_rgb_masks_decode_classes_and_void_exactly():
    import io
    array=np.array([[[0,0,0],[128,0,0],[192,128,128],[224,224,192]]],dtype=np.uint8)
    b=io.BytesIO();Image.fromarray(array).save(b,format='PNG')
    assert decode_mask(b.getvalue()).tolist()==[[0,1,15,255]]


def test_anchor_removal_cannot_discard_numbers():
    assert neutral_fields({'registered_read_text':'PERSON | TEMP 5 C | LIMIT 8 C'})=='TEMP 5 C | LIMIT 8 C'
    with pytest.raises(ValueError):neutral_fields({'registered_read_text':'TEMP 5 C | LIMIT 8 C'})


def test_options_are_counterbalanced_without_changing_question():
    q=assessment('Assume equality.',{'A':'absent','B':'consistent','C':'inconsistent'})
    assert '(A) No record is supplied.' in q and '(B) Internally consistent.' in q


def test_swapping_objects_leaves_all_record_pixels_identical():
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    from pathlib import Path
    if not Path(font).exists():pytest.skip('Linux integration font')
    left=Image.new('RGB',(100,160),'red');right=Image.new('RGB',(160,100),'blue')
    a=np.asarray(card_image(left,right,'VALUE 10 | LIMIT 20','VALUE 30 | LIMIT 20',font))
    b=np.asarray(card_image(right,left,'VALUE 10 | LIMIT 20','VALUE 30 | LIMIT 20',font))
    assert not np.array_equal(a[:370],b[:370])
    assert np.array_equal(a[370:],b[370:])
