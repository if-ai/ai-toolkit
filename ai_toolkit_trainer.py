import os
import sys
import pathlib
import uuid
import yaml
from typing import Dict, Any, Tuple

import folder_paths
from comfy.comfy_types.node_typing import IO

# ------------------------------------------------------------------
# Ensure local "toolkit" package (inside custom_nodes/ai-toolkit) is
# importable when ComfyUI loads this node.
# ------------------------------------------------------------------
_this_dir = pathlib.Path(__file__).resolve().parent
_parent_dir = str(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from toolkit.job import get_job  # noqa: E402


class AIToolkitTrainer:
    """Train Flux-Kontext, WAN-2.1, etc. from inside ComfyUI using AI-Toolkit.

    Workflow: the user places an example YAML in ComfyUI *input/*, selects it, picks
    a dataset subfolder also under *input/*, tweaks a few core hyper-parameters, and
    this node generates a temporary YAML, then calls `toolkit.job.get_job(...).run()`.
    """

    CATEGORY = "ai_toolkit"
    RETURN_TYPES = (IO.STRING,)  # returns output folder
    RETURN_NAMES = ("result",)
    FUNCTION = "train"

    @classmethod
    def INPUT_TYPES(cls):
        # Candidate YAML files in input/
        yaml_files = folder_paths.filter_files_extensions(
            os.listdir(folder_paths.get_input_directory()), [".yaml", ".yml"]
        )
        return {
            "required": {
                "base_yaml": (
                    yaml_files,
                    {"tooltip": "Base YAML config copied from ai-toolkit examples."},
                ),
                "dataset_subfolder": (
                    folder_paths.get_input_subfolders(),
                    {"default": "", "tooltip": "Sub-folder under input/ holding images & captions."},
                ),
                "output_subfolder": (
                    IO.STRING,
                    {"default": "ai_toolkit", "tooltip": "Dir inside output/ where checkpoints & samples go."},
                ),
                "steps_override": (
                    IO.INT,
                    {"default": 0, "min": 0, "max": 100_000, "step": 50, "tooltip": "0 = keep YAML value."},
                ),
                "lr_override": (
                    IO.FLOAT,
                    {"default": 0.0, "min": 1e-6, "max": 1e-2, "step": 1e-6, "tooltip": "0 = keep YAML value."},
                ),
                "push_to_hub": (
                    ["auto", "yes", "no"],
                    {"default": "auto", "tooltip": "Override push_to_hub flag."},
                ),
            },
            "optional": {
                "extra_yaml": (
                    IO.STRING,
                    {"multiline": True, "default": "", "tooltip": "Additional YAML snippet to merge."},
                )
            },
        }

    # -------------------- helpers --------------------
    @staticmethod
    def _merge(dst: Dict[str, Any], src: Dict[str, Any]):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                AIToolkitTrainer._merge(dst[k], v)
            else:
                dst[k] = v

    # -------------------- main --------------------
    def train(
        self,
        base_yaml: str,
        dataset_subfolder: str,
        output_subfolder: str,
        steps_override: int,
        lr_override: float,
        push_to_hub: str,
        extra_yaml: str = "",
    ) -> Tuple[str]:
        # 1. load yaml
        base_yaml_path = folder_paths.get_annotated_filepath(base_yaml, folder_paths.get_input_directory())
        if not os.path.isfile(base_yaml_path):
            raise FileNotFoundError(f"Base YAML not found: {base_yaml}")
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 2. apply overrides
        process = cfg["config"]["process"][0]
        if dataset_subfolder:
            process["datasets"][0]["folder_path"] = os.path.join(folder_paths.get_input_directory(), dataset_subfolder)
        out_dir = os.path.join(folder_paths.get_output_directory(), output_subfolder)
        process["training_folder"] = out_dir
        if steps_override > 0:
            process["train"]["steps"] = int(steps_override)
        if lr_override > 0:
            process["train"]["lr"] = float(lr_override)
        if push_to_hub != "auto":
            process["save"]["push_to_hub"] = (push_to_hub == "yes")
        if extra_yaml.strip():
            self._merge(process, yaml.safe_load(extra_yaml))

        # 3. write temp yaml
        tmp_yaml = os.path.join(folder_paths.get_temp_directory(), f"aitk_{uuid.uuid4().hex}.yaml")
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        os.makedirs(out_dir, exist_ok=True)

        # 4. run training (blocking)
        job = get_job(tmp_yaml)
        job.run()
        job.cleanup()

        return (out_dir,)


# -------------------- ComfyUI glue --------------------
NODE_CLASS_MAPPINGS = {"AI-Toolkit Trainer": AIToolkitTrainer}
NODE_DISPLAY_NAME_MAPPINGS = {"AI-Toolkit Trainer": "AI-Toolkit Trainer"} 