import argparse
import os
import sys
import toml
from scheduler.scheduler import Scheduler


def load_config(model_path=None):
    """Load config from model_path/efmnode.toml if available, else default config.toml.

    When model_path is provided:
      - Use <model_path>/efmnode.toml if it exists
      - Otherwise fall back to default config.toml with a warning
      - Always override ckpt_dir to point to model_path
    """
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")

    if model_path is not None:
        model_config_path = os.path.join(model_path, "efmnode.toml")
        if os.path.isfile(model_config_path):
            print(f"[INFO] Loading config from: {model_config_path}")
            config = toml.load(model_config_path)
        else:
            print(f"[WARNING] {model_config_path} not found, falling back to default config.toml", file=sys.stderr)
            config = toml.load(default_config_path)

        config.setdefault("model", {})
        config["model"]["ckpt_dir"] = model_path
        print(f"[INFO] Model checkpoint dir: {model_path}")
    else:
        config = toml.load(default_config_path)

    return config


def main():
    parser = argparse.ArgumentParser(description="EFMNode inference client")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Absolute path to model directory (overrides ckpt_dir in config)")
    args = parser.parse_args()

    config = load_config(args.model_path)
    scheduler = Scheduler(config)
    scheduler.run()


if __name__ == "__main__":
    main()
