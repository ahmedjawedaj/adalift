from __future__ import annotations

import argparse

from adalift.config import ModelConfig
from adalift.evaluation.profile import profile_parameters
from adalift.models.lift import LiFT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--prelude-depth", type=int, default=4)
    parser.add_argument("--recurrent-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=12)
    args = parser.parse_args()
    cfg = ModelConfig(
        hidden_size=args.hidden_size,
        prelude_depth=args.prelude_depth,
        recurrent_depth=args.recurrent_depth,
        num_heads=args.num_heads,
    )
    model = LiFT(cfg)
    profile = profile_parameters(model)
    print(f"total_parameters={profile.total_parameters:,}")
    print(f"trainable_parameters={profile.trainable_parameters:,}")


if __name__ == "__main__":
    main()
