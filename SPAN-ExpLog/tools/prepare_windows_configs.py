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

SCREENING_10K = {
    # A was run before checkpoint frequency was aligned with validation.
    'train_A_baseline_x4_10k.yml': ('round2_A_baseline_x4_10k', 'A_baseline', 5e-4, 5000),
    'train_C_matched_b3_x4_10k.yml': ('round2_C_matched_b3_x4_10k', 'C_matched_b3', 5e-4, 2000),
    'train_D_explog_b3_x4_10k.yml': ('round2_D_explog_b3_x4_10k', 'D_explog_b3', 5e-4, 2000),
    'train_D_explog_b3_x4_10k_lr2p5e4.yml': ('round3_D_explog_b3_x4_10k_lr2p5e4', 'D_explog_b3', 2.5e-4, 2000),
    'train_D_explog_b3_x4_10k_lr3p5e4.yml': ('round3_D_explog_b3_x4_10k_lr3p5e4', 'D_explog_b3', 3.5e-4, 2000),
}


def posix(path):
    return Path(path).as_posix()


def make_config(template, label, dataset_root):
    config = copy.deepcopy(template)
    changes = VARIANTS[label]
    config['datasets']['train']['dataroot_gt'] = f'{posix(dataset_root)}/DIV2K/DIV2K_train_HR'
    config['datasets']['train']['dataroot_lq'] = f'{posix(dataset_root)}/DIV2K/DIV2K_train_LR_bicubic/X4'
    config['datasets']['val']['dataroot_gt'] = f'{posix(dataset_root)}/DIV2K/DIV2K_valid_HR'
    config['datasets']['val']['dataroot_lq'] = f'{posix(dataset_root)}/DIV2K/DIV2K_valid_LR_bicubic/X4'
    config['network_g'].update(changes)
    if changes['type'] == 'SPAN':
        for key in ('enhance_blocks', 'enhancer', 'explog_groups', 'explog_alpha'):
            config['network_g'].pop(key, None)
    config['train']['ema_decay'] = 0
    return config


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
    for label in VARIANTS:
        config = make_config(template, label, args.dataset_root)
        config['name'] = f'smoke_{label}_x4'
        config['train']['total_iter'] = args.smoke_iters
        # Short smoke runs must validate the current net_g. EMA=0.999 lags far
        # behind at 1,000 iterations and produced misleadingly low metrics.
        config['val']['val_freq'] = max(1, args.smoke_iters // 2)
        config['logger']['save_checkpoint_freq'] = max(1, args.smoke_iters // 2)
        with (option_dir / f'train_{label}_x4_smoke.yml').open('w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    for filename, (name, label, learning_rate, checkpoint_frequency) in SCREENING_10K.items():
        config = make_config(template, label, args.dataset_root)
        config['name'] = name
        config['train']['optim_g']['lr'] = learning_rate
        config['train']['total_iter'] = 10000
        config['val']['val_freq'] = 2000
        config['logger']['save_checkpoint_freq'] = checkpoint_frequency
        with (option_dir / filename).open('w', encoding='utf-8') as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    print(
        f'Generated baseline test, {len(VARIANTS)} smoke configs, and '
        f'{len(SCREENING_10K)} validated 10k configs in {option_dir}'
    )


if __name__ == '__main__':
    main()
