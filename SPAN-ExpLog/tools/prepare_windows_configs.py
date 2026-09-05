"""Generate Windows experiment YAML files from one pair of root paths."""

import argparse
import copy
from pathlib import Path

import yaml


DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
VARIANTS = {
    'A_baseline': {'type': 'SPAN'},
    'B_gn_b3': {'type': 'SPANExpLog', 'enhance_blocks': [3], 'enhancer': 'gn'},
    'C_matched_b3': {'type': 'SPANExpLog', 'enhance_blocks': [3], 'enhancer': 'matched_conv'},
    'D_explog_b3': {'type': 'SPANExpLog', 'enhance_blocks': [3], 'enhancer': 'explog'},
    'E_explog_b246': {'type': 'SPANExpLog', 'enhance_blocks': [2, 4, 6], 'enhancer': 'explog'},
    'F_explog_all': {'type': 'SPANExpLog', 'enhance_blocks': [1, 2, 3, 4, 5, 6], 'enhancer': 'explog'},
}


def posix(path):
    return Path(path).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, help='Example: D:/datasets')
    parser.add_argument('--checkpoint', required=True, help='Official SPAN x4 checkpoint')
    parser.add_argument('--smoke-iters', type=int, default=1000)
    args = parser.parse_args()

    option_dir = Path('options/experiments/generated')
    option_dir.mkdir(parents=True, exist_ok=True)
    with Path('options/experiments/test_SPAN_x4_template.yml').open(encoding='utf-8') as handle:
        test = yaml.safe_load(handle)
    for item in test['datasets'].values():
        dataset = item['name']
        item['dataroot_gt'] = f'{posix(args.dataset_root)}/SR/{dataset}/GTmod12'
        item['dataroot_lq'] = f'{posix(args.dataset_root)}/SR/{dataset}/LRbicx4'
    test['path']['pretrain_network_g'] = posix(args.checkpoint)
    with (option_dir / 'test_SPAN_x4_windows.yml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(test, handle, sort_keys=False, allow_unicode=True)

    with Path('options/experiments/train_SPAN_ExpLog_x4_template.yml').open(encoding='utf-8') as handle:
        template = yaml.safe_load(handle)
    for label, changes in VARIANTS.items():
        config = copy.deepcopy(template)
        config['name'] = f'smoke_{label}_x4'
        config['datasets']['train']['dataroot_gt'] = f'{posix(args.dataset_root)}/DIV2K/DIV2K_train_HR'
        config['datasets']['train']['dataroot_lq'] = f'{posix(args.dataset_root)}/DIV2K/DIV2K_train_LR_bicubic/X4'
        config['datasets']['val']['dataroot_gt'] = f'{posix(args.dataset_root)}/DIV2K/DIV2K_valid_HR'
        config['datasets']['val']['dataroot_lq'] = f'{posix(args.dataset_root)}/DIV2K/DIV2K_valid_LR_bicubic/X4'
        config['network_g'].update(changes)
        if changes['type'] == 'SPAN':
            for key in ('enhance_blocks', 'enhancer', 'explog_groups', 'explog_alpha'):
                config['network_g'].pop(key, None)
        config['train']['total_iter'] = args.smoke_iters
        # Short smoke runs must validate the current net_g. EMA=0.999 lags far
        # behind at 1,000 iterations and produced misleadingly low metrics.
        config['train']['ema_decay'] = 0
        config['val']['val_freq'] = max(1, args.smoke_iters // 2)
        config['logger']['save_checkpoint_freq'] = max(1, args.smoke_iters // 2)
        with (option_dir / f'train_{label}_x4_smoke.yml').open('w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    print(f'Generated baseline test and {len(VARIANTS)} smoke-training configs in {option_dir}')


if __name__ == '__main__':
    main()
