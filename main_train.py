# This python script functions as `allennlp train` commond in `allennlp` setup
# e.g.  allennlp train experiments/bi_sst/embedding--glove__lstm.json -s model/bi_sst/embedding--glove__lstm/  --include-package my_library 

import argparse
from allennlp import commands
from allennlp.common.util import import_module_and_submodules
from attack_utils import model_util
import sys
sys.dont_write_bytecode = True

def parse_train_args():
    # from https://github.com/allenai/allennlp/blob/5338bd8b4a7492e003528fe607210d2acc2219f5/allennlp/commands/train.py
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "param_path", 
        type=str, 
        # default='experiments/ag_news/cnn.json',
        help="path to parameter file describing the model to be trained"
    )

    parser.add_argument(
        "-s",
        "--serialization-dir",
        type=str,
        default='models/ag_news/debug',
        help="directory in which to save the model and its logs",
    )

    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        default=False,
        help="recover training from the state in serialization_dir",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        required=False,
        help="overwrite the output directory if it exists",
    )

    parser.add_argument(
        "-o",
        "--overrides",
        type=str,
        default="",
        help=(
            "a json(net) structure used to override the experiment configuration, e.g., "
            "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
            " with nested dictionaries or with dot syntax."
        ),
    )

    parser.add_argument(
        "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "do not train a model, but create a vocabulary, show dataset statistics and "
            "other training information"
        ),
    )
    parser.add_argument(
        "--file-friendly-logging",
        action="store_true",
        default=False,
        help="outputs tqdm status on separate lines and slows tqdm refresh rate",
    )

    parser.add_argument(
                    "--include-package",
                    type=str,
                    action="append",
                    default=[],
                    help="additional packages to include",
                )


    # Now we can parse the arguments.
    args = parser.parse_args()

    return parser, args

def main():
    parser, args = parse_train_args()

    # Import any additional modules needed (to register custom classes).
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)



    commands.train.train_model_from_file(
            parameter_filename=args.param_path,
            serialization_dir=args.serialization_dir,
            overrides=args.overrides,
            recover=args.recover,
            force=args.force,
            node_rank=args.node_rank,
            include_package=args.include_package,
            dry_run=args.dry_run,
            file_friendly_logging=args.file_friendly_logging,
        )
    

if __name__=="__main__":
    main()

