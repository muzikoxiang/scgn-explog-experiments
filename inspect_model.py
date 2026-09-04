"""Print parameter count and ptflops MACs for one reconstructed variant."""

import argparse

from models import VARIANTS, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    args = parser.parse_args()
    model = build_model(args.variant)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"variant={args.variant} params={parameters}")
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("ptflops is not installed; skipping MAC/FLOP estimation")
        return
    macs, params = get_model_complexity_info(
        model, (1, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False
    )
    print(f"MACs/FLOPs={macs} ptflops_params={params}")


if __name__ == "__main__":
    main()

