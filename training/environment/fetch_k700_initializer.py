"""Fetch and audit the documented upstream K700 initializer; no GPU training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

SOURCE_URL = 'https://drive.usercontent.google.com/download?id=1wn1392Kn6CFxcSH6lJpqZky9-PJxqTlY&export=download&confirm=t'
SOURCE_DOC = 'https://github.com/JunweiLiang/aicity_action/blob/5315a09e441aaba69aca68bcaf13607a7b7c3e39/README.md'
EXPECTED_BYTES = 423231999
EXPECTED_CFG_SHA256 = '1f2bad063e38dcd590ea882e85c8229bee24fb27bf84983aabccbee82e8c4e1e'
EXPECTED_SHAPES_SHA256 = 'd779cc3df40e1bec8a86506458cbfa3d0d8bce3c7747b0ce4ce34382dcf41b1d'


def audit_metadata(state):
    import yaml
    cfg_text = state.get('cfg')
    if not isinstance(cfg_text, str) or hashlib.sha256(cfg_text.encode()).hexdigest() != EXPECTED_CFG_SHA256:
        raise ValueError('Downloaded checkpoint configuration differs from the inspected public source')
    tensors = state.get('model_state', {})
    shapes = {key: list(value.shape) for key, value in tensors.items()}
    fingerprint = hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if fingerprint != EXPECTED_SHAPES_SHA256:
        raise ValueError('Downloaded checkpoint tensor metadata differs from the inspected public source')
    cfg = yaml.safe_load(cfg_text)
    if cfg['TRAIN']['DATASET'] != 'kinetics' or cfg['MODEL']['NUM_CLASSES'] != 700:
        raise ValueError('Expected a generic Kinetics-700 model')
    if shapes.get('head.projection.weight') != [700, 768] or state.get('epoch') != 199:
        raise ValueError('Unexpected classification head or pretraining epoch')
    if cfg['TRAIN'].get('CHECKPOINT_FILE_PATH'):
        raise ValueError('Unexpected previous-checkpoint training history')
    return dict(training_dataset='Kinetics-700', classes=700, pretraining_epochs=200,
                source_config_sha256=EXPECTED_CFG_SHA256, tensor_shapes_sha256=fingerprint,
                head_shape=shapes['head.projection.weight'],
                saved_crop_size=cfg['DATA']['TRAIN_CROP_SIZE'],
                spatial_position_shape=shapes['pos_embed_spatial'],
                geometry_note='Saved config reports crop 224; the supplied spatial-position tensor is 112x112, compatible with crop 448 at patch stride 4. Full model shape matching is mandatory.')


def fetch(path):
    if path.exists():
        if path.stat().st_size != EXPECTED_BYTES:
            raise ValueError('Existing raw checkpoint has the wrong size; preserved for inspection')
        print('Raw checkpoint already downloaded; auditing it.', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.part')
    start = partial.stat().st_size if partial.exists() else 0
    if start > EXPECTED_BYTES:
        raise ValueError('Oversized partial download; preserved for inspection')
    if start < EXPECTED_BYTES:
        headers = {'User-Agent': 'Mozilla/5.0', 'Cache-Control': 'no-cache'}
        if start:
            headers['Range'] = f'bytes={start}-'
        request = urllib.request.Request(SOURCE_URL, headers=headers)
        print(f'Downloading public K700 checkpoint: {EXPECTED_BYTES/1e6:.1f} MB', flush=True)
        with urllib.request.urlopen(request, timeout=120) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError('Google Drive returned a webpage instead of the checkpoint')
            if start and response.status != 206:
                start = 0
            if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                raise ValueError('Invalid resume range from Google Drive')
            received, last_print = start, time.monotonic()
            with partial.open('ab' if start else 'wb') as target:
                for chunk in iter(lambda: response.read(4*1024*1024), b''):
                    target.write(chunk)
                    received += len(chunk)
                    if received > EXPECTED_BYTES:
                        raise ValueError('Download is larger than the inspected source')
                    if time.monotonic()-last_print > 5:
                        print(f'  {100*received/EXPECTED_BYTES:.1f}%', flush=True)
                        last_print = time.monotonic()
    if partial.stat().st_size != EXPECTED_BYTES:
        raise ValueError('Incomplete download; rerun this helper to resume')
    partial.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', default='prepared/server')
    p.add_argument('--project-root', default=str(Path.cwd()))
    a = p.parse_args()
    root = Path(a.project_root).resolve()
    if not (root/'scripts/prepare_initializer.py').is_file():
        raise ValueError('Run from the m2dar-training directory or provide --project-root')
    sys.path.insert(0, str(root/'scripts'))
    from common import load_torch_checkpoint, sha256, verify_bundle, write_json
    from prepare_initializer import prepare
    bundle = Path(a.bundle)
    if not bundle.is_absolute():
        bundle = root/bundle
    dataset = verify_bundle(bundle)
    source = root/'checkpoints/original/k700_train_mvitV2_full_16x4_fromscratch_e200_448.pyth'
    output = root/'checkpoints/generic_initializer.pyth'
    if output.exists():
        prepared = load_torch_checkpoint(output).get('preparation', {})
        if prepared.get('purpose') != 'generic_initializer' or prepared.get('dataset_fingerprint') != dataset['dataset_fingerprint']:
            raise ValueError('Existing initializer belongs to a different experiment; preserved')
        print('Prepared initializer already exists for this dataset:', output)
        return
    fetch(source)
    raw_sha = sha256(source)
    print('Raw checkpoint SHA256:', raw_sha, flush=True)
    state = load_torch_checkpoint(source)
    audit = audit_metadata(state)
    del state
    audit.update(raw_sha256=raw_sha, source_url=SOURCE_URL, source_documentation=SOURCE_DOC,
                 m2dar_pretrained_link_status='HTTP 404 at preparation time',
                 public_architecture_config_matches_m2dar=True,
                 provenance_limit='Assessment uses the upstream project documentation and embedded training metadata, not an independent reconstruction of the original pretraining data.')
    write_json(root/'checks/k700_source_audit.json', audit)
    provenance = dict(kind='generic_action_pretraining', source_url=SOURCE_URL,
                      training_datasets=['Kinetics-700'], aicity_or_syndd2_exposure=False, sha256=raw_sha,
                      evidence='The pinned upstream README identifies this download as the generic K700 initializer, separately from its AiCity-finetuned weights. Its saved TRAIN.DATASET is kinetics, MODEL.NUM_CLASSES is 700, the head has shape [700,768], epoch is 199, and TRAIN.CHECKPOINT_FILE_PATH is empty. Full configuration and shape fingerprints match the inspected public artifact. See checks/k700_source_audit.json for provenance limitations and the saved-crop versus position-grid discrepancy.',
                      evidence_url=SOURCE_DOC)
    provenance_path = root/'checkpoints/k700_provenance.json'
    write_json(provenance_path, provenance)
    print('K700 source metadata passed. Checking every backbone tensor against M2DAR on CPU...', flush=True)
    prepare(bundle, source, provenance_path, output)
    print('READY: generic initializer prepared. No GPU training has started.', flush=True)


if __name__ == '__main__':
    main()
