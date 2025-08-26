import os
import io
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import requests

# Import centralized logging setup
try:
    from ..utils.debug_utils import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Optional Comfy file resolver (works in ComfyDeploy)
try:
    from folder_paths import get_annotated_filepath  # type: ignore
except Exception:
    def get_annotated_filepath(path: str) -> str:  # fallback no-op
        return path

# Import EXR utilities
try:
    from ..utils.exr_utils import ExrProcessor
except ImportError:
    raise ImportError("EXR utilities are required but not available. Please ensure utils are properly installed.")

def _linear_to_srgb_inplace(t: torch.Tensor) -> None:
    """
    Convert linear RGB to sRGB in-place for tensors in [B, H, W, 3] or [H, W, 3].
    This mirrors the External EXR node behavior.
    """
    # Works on last channel dim
    less = t <= 0.0031308
    t[less] = t[less] * 12.92
    t[~less] = torch.pow(t[~less].clamp(min=0.0), 1.0 / 2.4) * 1.055 - 0.055


def _apply_exposure(rgb: torch.Tensor, exposure_stops: float) -> torch.Tensor:
    """
    Apply exposure adjustment in stops (powers of 2).
    Expects and returns [1, H, W, 3] float32 tensor.
    """
    if rgb is None or exposure_stops == 0.0:
        return rgb
    exposure_multiplier = 2.0 ** exposure_stops
    return rgb * exposure_multiplier


def _apply_tonemap(rgb: torch.Tensor, tonemap: str, exposure_stops: float = 0.0) -> torch.Tensor:
    """
    Apply exposure and tonemapping to HDR/EXR data.
    Expects and returns [1, H, W, 3] float32 tensor.
    """
    if rgb is None:
        return rgb
    
    # Apply exposure first
    out = _apply_exposure(rgb, exposure_stops)
    
    if tonemap == "sRGB":
        _linear_to_srgb_inplace(out)
        return out.clamp(0.0, 1.0)
    elif tonemap == "Reinhard":
        out = out.clamp(min=0.0)
        out = out / (out + 1.0)
        _linear_to_srgb_inplace(out)
        return out.clamp(0.0, 1.0)
    elif tonemap == "ACES":
        # Simple ACES-like tone mapping
        out = out.clamp(min=0.0)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        out = (out * (a * out + b)) / (out * (c * out + d) + e)
        return out.clamp(0.0, 1.0)
    # "linear" or anything else: return with exposure applied but no tone curve
    return out


class LoadExr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "path/to/image.exr",
                    "description": "Full path to the EXR file"
                }),
                "normalize": ("BOOLEAN", {
                    "default": False,
                    "description": "Normalize image values to the 0-1 range"
                })
            },
            "optional": {
                # ComfyDeploy External EXR-compatible inputs
                "input_id": ("STRING", {"multiline": False, "default": "input_exr"}),
                "exr_file": ("STRING", {"default": ""}),  # URL or local path
                "tonemap": (["linear", "sRGB", "Reinhard", "ACES"], {"default": "linear"}),
                "exposure_stops": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "Exposure adjustment in stops (powers of 2)"
                }),
                "default_image": ("IMAGE",),
                "default_mask": ("MASK",),
                "display_name": ("STRING", {"multiline": False, "default": ""}),
                "description": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "layer_data": "DICT"
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CRYPTOMATTE", "LAYERS", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "alpha", "cryptomatte", "layers", "layer names", "raw layer info", "metadata")
    
    FUNCTION = "load_image"
    CATEGORY = "Image/EXR"
    
    @classmethod
    def IS_CHANGED(cls, image_path, normalize=False, **kwargs):
        """
        Smart caching based on file modification time and size.
        Only reload if file actually changed or parameters changed.
        """
        try:
            exr_file = kwargs.get("exr_file", "")
            tonemap = kwargs.get("tonemap", "sRGB")
            # If using URL exr_file, we can't reliably stat; force execute
            if exr_file and (exr_file.startswith("http://") or exr_file.startswith("https://")):
                return float("NaN")
            # If using local exr_file, base cache on that path instead
            if exr_file and os.path.isfile(get_annotated_filepath(exr_file)):
                path = get_annotated_filepath(exr_file)
                stat = os.stat(path)
                return f"{path}_{stat.st_mtime}_{stat.st_size}_{normalize}_{tonemap}_{kwargs.get('exposure_stops', 0.0)}"
            if not os.path.isfile(image_path):
                return float("NaN")  # File doesn't exist, always try to load
            
            stat = os.stat(image_path)
            # Create hash from file path, modification time, size, and normalize parameter
            return f"{image_path}_{stat.st_mtime}_{stat.st_size}_{normalize}_{tonemap}_{kwargs.get('exposure_stops', 0.0)}"
        except Exception:
            # If we can't access file info, always try to load
            return float("NaN")
    
    def load_image(self, image_path: str, normalize: bool = False,
                   node_id: Optional[str] = None, layer_data: Optional[Dict[str, Any]] = None,
                   **kwargs) -> List:
        """
        Load a single EXR image with support for multiple layers/channel groups.
        Returns:
        - Base RGB image tensor (image)
        - Alpha channel tensor (alpha)
        - Dictionary of all cryptomatte layers as tensors (cryptomatte)
        - Dictionary of all non-cryptomatte layers as tensors (layers)
        - List of processed layer names matching keys in the returned dictionaries (layer names)
        - List of raw channel names from the file (raw layer info)
        - Metadata as JSON string (metadata)
        """
        # Check for OIIO availability
        ExrProcessor.check_oiio_availability()

        # External EXR compatible params
        exr_file: str = kwargs.get("exr_file", "") or ""
        tonemap: str = kwargs.get("tonemap", "linear")
        exposure_stops: float = kwargs.get("exposure_stops", 0.0)
        default_image = kwargs.get("default_image", None)
        default_mask = kwargs.get("default_mask", None)
        input_id: str = kwargs.get("input_id", "input_exr")
        display_name: str = kwargs.get("display_name", "")
        description: str = kwargs.get("description", "")

        temp_path: Optional[str] = None
        try:
            # Decide source path
            source_path = image_path
            used_external = False
            if exr_file and len(exr_file.strip()) > 0:
                used_external = True
                if exr_file.startswith(("http://", "https://")):
                    # Download to temp file so OIIO can read and we keep layer support
                    response = requests.get(exr_file)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
                        tmp.write(response.content)
                        temp_path = tmp.name
                    source_path = temp_path
                else:
                    # Resolve local path via Comfy's folder_paths if available
                    source_path = get_annotated_filepath(exr_file)

            # Validate path or fallback to defaults
            if not source_path or not os.path.isfile(source_path):
                if default_image is not None or default_mask is not None:
                    # Build External EXR-style default outputs but match our return signature
                    metadata = {
                        "source": "default_fallback",
                        "input_id": input_id,
                        "display_name": display_name,
                        "description": description,
                        "tonemap": tonemap,
                        "exposure_stops": exposure_stops,
                        "external_used": used_external,
                    }
                    return [
                        default_image if default_image is not None else torch.zeros((1, 1, 1, 3)),
                        default_mask if default_mask is not None else torch.zeros((1, 1, 1)),
                        {},  # cryptomatte
                        {},  # layers
                        [],  # processed layer names
                        [],  # raw channel names
                        str(metadata),
                    ]
                # No defaults; treat as normal error
                raise FileNotFoundError(f"Image not found: {source_path}")

            # Process EXR with full layer support
            result = ExrProcessor.process_exr_data(source_path, normalize, node_id, layer_data)

            # Handle result with or without preview wrapper
            if isinstance(result, dict):
                rgb_tensor, alpha_tensor, cryptomatte, layers, processed_names, raw_names, metadata_json = result["result"]
                ui_part = result.get("ui")
            else:
                rgb_tensor, alpha_tensor, cryptomatte, layers, processed_names, raw_names, metadata_json = result
                ui_part = None

            # Apply exposure and tone mapping to base RGB (does not affect individual layers)
            # Always apply tone mapping for consistency, regardless of source
            rgb_tensor = _apply_tonemap(rgb_tensor, tonemap, exposure_stops)

            # Enrich metadata with external fields
            try:
                import json  # local import to avoid top-level clash
                metadata: Dict[str, Any] = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                metadata = {"file_path": source_path}

            metadata.update({
                "comfydeploy": {
                    "input_id": input_id,
                    "display_name": display_name,
                    "description": description,
                    "tonemap": tonemap,
                    "exposure_stops": exposure_stops,
                    "external_used": used_external,
                    "exr_file_param": exr_file,
                }
            })

            final_result = [
                rgb_tensor,
                alpha_tensor,
                cryptomatte,
                layers,
                processed_names,
                raw_names,
                __import__("json").dumps(metadata),
            ]

            if ui_part is not None:
                return {"ui": ui_part, "result": final_result}
            return final_result

        except Exception as e:
            # External-EXR-style graceful fallback when exr_file path was provided
            if (exr_file and (default_image is not None or default_mask is not None)):
                logger.error(f"Error loading External EXR '{exr_file}': {str(e)}")
                metadata = {
                    "source": "error_fallback",
                    "error": str(e),
                    "input_id": input_id,
                    "tonemap": tonemap,
                    "exposure_stops": exposure_stops,
                }
                import json as _json
                return [
                    default_image if default_image is not None else torch.zeros((1, 1, 1, 3)),
                    default_mask if default_mask is not None else torch.zeros((1, 1, 1)),
                    {},
                    {},
                    [],
                    [],
                    _json.dumps(metadata),
                ]
            # Preserve original behavior for standard local loads
            logger.error(f"Error loading EXR file {image_path}: {str(e)}")
            raise
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass












