import datetime
import hashlib
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import folder_paths
import node_helpers
from comfy.cli_args import args


CATEGORY = "metadata"


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return {
            "__type__": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _pretty_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)


def _parse_json_maybe(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _sanitize_large_values(value, max_string_length=50000):
    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return {
            "__omitted_large_string__": True,
            "length": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(),
            "preview": value[:500],
        }
    if isinstance(value, list):
        return [_sanitize_large_values(item, max_string_length=max_string_length) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_large_values(item, max_string_length=max_string_length)
            for key, item in value.items()
        }
    return value


def _node(prompt, node_id):
    if not isinstance(prompt, dict) or node_id is None:
        return None
    return prompt.get(str(node_id))


def _linked_node_id(value):
    if isinstance(value, list) and len(value) >= 1:
        return str(value[0])
    return None


def _linked_node_ids(value):
    node_ids = []
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int):
            node_ids.append(str(value[0]))
        else:
            for item in value:
                node_ids.extend(_linked_node_ids(item))
    elif isinstance(value, dict):
        for item in value.values():
            node_ids.extend(_linked_node_ids(item))
    return node_ids


def _is_output_root(class_type):
    return class_type in {
        "MSV_SaveImageWithMetadataJson",
        "SaveImage",
        "PreviewImage",
        "SaveImageWebsocket",
    }


def _used_node_ids(prompt):
    if not isinstance(prompt, dict):
        return set()

    roots = []
    sampler_roots = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        if _is_output_root(class_type):
            roots.append(str(node_id))
        if class_type in {"KSampler", "KSamplerAdvanced"}:
            sampler_roots.append(str(node_id))

    if not roots:
        roots = sampler_roots

    used = set()
    stack = list(roots)
    while stack:
        node_id = stack.pop()
        if node_id in used:
            continue
        used.add(node_id)
        node = prompt.get(str(node_id))
        if not isinstance(node, dict):
            continue
        for input_value in node.get("inputs", {}).values():
            for linked_id in _linked_node_ids(input_value):
                if linked_id not in used:
                    stack.append(linked_id)

    return used


def _text_from_conditioning(prompt, link_value):
    node_id = _linked_node_id(link_value)
    seen = set()

    while node_id and node_id not in seen:
        seen.add(node_id)
        node = _node(prompt, node_id)
        if not isinstance(node, dict):
            return None

        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if class_type in {"CLIPTextEncode", "ImpactWildcardEncode"} and "text" in inputs:
            return inputs.get("text")

        # Follow common passthrough/combine nodes only when there is a single likely conditioning input.
        conditioning_links = [
            value for key, value in inputs.items()
            if "conditioning" in key.lower() or key.lower() in {"positive", "negative"}
        ]
        if len(conditioning_links) != 1:
            return None
        node_id = _linked_node_id(conditioning_links[0])

    return None


def _parse_float_list(text):
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[,;\n\t ]+", str(text)) if part.strip()]
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            pass
    return values


def _build_range_values(start, stop, step, direction):
    step_abs = abs(float(step or 0))
    if step_abs <= 0:
        return []

    lo = min(float(start), float(stop))
    hi = max(float(start), float(stop))
    values = []

    if direction == "decrement":
        current = hi
        while current >= lo - 1e-12:
            values.append(current)
            current -= step_abs
    else:
        current = lo
        while current <= hi + 1e-12:
            values.append(current)
            current += step_abs

    return values


def _format_float(value):
    if value is None:
        return None
    return float(f"{float(value):.6g}")


def _convert_datetime_format(format_text):
    replacements = (
        ("yyyy", "%Y"),
        ("YYYY", "%Y"),
        ("yy", "%y"),
        ("YY", "%y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("DD", "%d"),
        ("HH", "%H"),
        ("hh", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    )
    converted = format_text
    for source, target in replacements:
        converted = converted.replace(source, target)
    return converted


def _expand_filename_prefix(filename_prefix):
    now = datetime.datetime.now()

    def replace_datetime(match):
        format_text = _convert_datetime_format(match.group(2))
        return now.strftime(format_text)

    filename_prefix = re.sub(r"%(date|time):([^%]+)%", replace_datetime, filename_prefix)
    filename_prefix = filename_prefix.replace("%date%", now.strftime("%Y-%m-%d"))
    filename_prefix = filename_prefix.replace("%time%", now.strftime("%H-%M-%S"))
    return filename_prefix


def _per_sample_lora_weight(inputs, image_index):
    mode = inputs.get("mode")
    if mode == "manual_values":
        values = _parse_float_list(inputs.get("manual_values", ""))
    elif mode == "range":
        values = _build_range_values(
            inputs.get("start", 0.0),
            inputs.get("stop", 0.0),
            inputs.get("step", 1.0),
            inputs.get("direction", "increment"),
        )
    else:
        values = []

    if not values:
        return None, None, 0

    index = int(image_index or 0)
    if index < len(values):
        return _format_float(values[index]), index, len(values)
    wrapped_index = index % len(values)
    return _format_float(values[wrapped_index]), wrapped_index, len(values)


def _usage_suffix(item):
    if isinstance(item, dict) and item.get("used") is False:
        return " (unused)"
    return ""


def _model_display_name(model):
    for key in ("ckpt_name", "unet_name", "clip_name", "model_name", "model_path", "config_name"):
        if model.get(key):
            return model[key]
    return model.get("class_type")


def _model_role(class_type):
    if class_type in {"CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader"}:
        return "clip"
    return "base"


def _collect_prompt_summary(prompt, image_index=None):
    summary = {
        "positive_prompt": None,
        "negative_prompt": None,
        "samplers": [],
        "loras": [],
        "models": [],
        "vae": [],
        "image_size": [],
        "node_count": 0,
    }

    if not isinstance(prompt, dict):
        return summary

    used_nodes = _used_node_ids(prompt)

    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue

        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        summary["node_count"] += 1

        if class_type in {"KSampler", "KSamplerAdvanced"}:
            sampler = {"node_id": node_id, "class_type": class_type}
            sampler["used"] = str(node_id) in used_nodes
            for key in (
                "seed",
                "noise_seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
                "start_at_step",
                "end_at_step",
                "add_noise",
                "return_with_leftover_noise",
            ):
                if key in inputs:
                    sampler[key] = inputs[key]

            positive = _text_from_conditioning(prompt, inputs.get("positive"))
            negative = _text_from_conditioning(prompt, inputs.get("negative"))
            if positive is not None:
                sampler["positive_prompt_source_node"] = _linked_node_id(inputs.get("positive"))
                if summary["positive_prompt"] is None:
                    summary["positive_prompt"] = positive
            if negative is not None:
                sampler["negative_prompt_source_node"] = _linked_node_id(inputs.get("negative"))
                if summary["negative_prompt"] is None:
                    summary["negative_prompt"] = negative
            summary["samplers"].append(sampler)

        if "LoraLoader" in class_type or "LoRA" in class_type or "LORA" in class_type:
            lora = {
                "node_id": node_id,
                "name": inputs.get("lora_name"),
                "used": str(node_id) in used_nodes,
            }
            if class_type == "PerSampleLoraLoader":
                weight, weight_index, weight_count = _per_sample_lora_weight(inputs, image_index)
                lora["weight"] = weight
                lora["weight_index"] = weight_index
                lora["weight_count"] = weight_count
                lora["sample_number"] = int(image_index or 0) + 1
            else:
                strength_model = inputs.get("strength_model")
                strength_clip = inputs.get("strength_clip")
                if strength_model == strength_clip or strength_clip is None:
                    lora["weight"] = _format_float(strength_model)
                else:
                    lora["weight_model"] = _format_float(strength_model)
                    lora["weight_clip"] = _format_float(strength_clip)
            summary["loras"].append(lora)

        if class_type in {
            "CheckpointLoaderSimple",
            "CheckpointLoader",
            "unCLIPCheckpointLoader",
            "UNETLoader",
            "CLIPLoader",
            "DualCLIPLoader",
            "TripleCLIPLoader",
            "DiffusersLoader",
        }:
            model = {
                "node_id": node_id,
                "class_type": class_type,
                "role": _model_role(class_type),
                "used": str(node_id) in used_nodes,
            }
            for key in (
                "ckpt_name",
                "config_name",
                "unet_name",
                "clip_name",
                "clip_name1",
                "clip_name2",
                "clip_name3",
                "model_name",
                "model_path",
            ):
                if key in inputs:
                    model[key] = inputs[key]
            summary["models"].append(model)

        if class_type == "VAELoader":
            summary["vae"].append({
                "node_id": node_id,
                "vae_name": inputs.get("vae_name"),
                "used": str(node_id) in used_nodes,
            })

        if class_type in {"EmptyLatentImage", "EmptySD3LatentImage"}:
            size = {"node_id": node_id, "used": str(node_id) in used_nodes}
            for key in ("width", "height", "batch_size"):
                if key in inputs:
                    size[key] = inputs[key]
            summary["image_size"].append(size)

    return summary


def _build_metadata(prompt, extra_pnginfo, filename=None, subfolder=None, image_index=None):
    extra_pnginfo = extra_pnginfo or {}
    prompt_summary = _collect_prompt_summary(prompt, image_index=image_index)
    positive_prompt = prompt_summary.pop("positive_prompt", None)
    negative_prompt = prompt_summary.pop("negative_prompt", None)
    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    sanitized_prompt = _sanitize_large_values(prompt)
    sanitized_workflow = _sanitize_large_values(workflow)
    extra_without_workflow = {
        key: value for key, value in extra_pnginfo.items()
        if key != "workflow"
    } if isinstance(extra_pnginfo, dict) else extra_pnginfo
    sanitized_extra_without_workflow = _sanitize_large_values(extra_without_workflow)

    metadata = {
        "format": "ComfyUI Metadata Saver Viewer",
        "format_version": 4,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": {
            "filename": filename,
            "subfolder": subfolder,
            "batch_index": image_index,
        },
        "summary": prompt_summary,
        "raw": {
            "prompt": sanitized_prompt,
            "workflow": sanitized_workflow,
            "extra_pnginfo": sanitized_extra_without_workflow,
        },
        "compaction": {
            "large_strings_over_chars": 50000,
            "replacement": "Large strings are replaced with length, sha256, and preview to avoid recursive metadata growth from preview/text display nodes.",
        },
    }

    # Fooocus-like convenience keys for quick reading, while keeping all raw data above.
    if positive_prompt:
        metadata["Prompt"] = positive_prompt
    if negative_prompt:
        metadata["Negative Prompt"] = negative_prompt
    if prompt_summary["samplers"]:
        sampler = prompt_summary["samplers"][0]
        metadata["Sampler"] = sampler.get("sampler_name")
        metadata["Scheduler"] = sampler.get("scheduler")
        metadata["Seed"] = sampler.get("seed", sampler.get("noise_seed"))
        metadata["Steps"] = sampler.get("steps")
        metadata["Guidance Scale"] = sampler.get("cfg")
        metadata["Denoise"] = sampler.get("denoise")
    base_models = [model for model in prompt_summary["models"] if model.get("role") == "base"]
    clip_models = [model for model in prompt_summary["models"] if model.get("role") == "clip"]
    if base_models:
        for index, model in enumerate(base_models, start=1):
            label = "Base Model" if index == 1 else f"Model {index}"
            metadata[label] = f"{_model_display_name(model)}{_usage_suffix(model)}"
    if clip_models:
        for index, model in enumerate(clip_models, start=1):
            label = "CLIP Model" if index == 1 else f"CLIP Model {index}"
            metadata[label] = f"{_model_display_name(model)}{_usage_suffix(model)}"
    if prompt_summary["vae"]:
        vae = prompt_summary["vae"][0]
        metadata["VAE"] = f"{vae.get('vae_name')}{_usage_suffix(vae)}"
    for index, lora in enumerate(prompt_summary["loras"], start=1):
        name = lora.get("name")
        suffix = _usage_suffix(lora)
        if lora.get("weight") is not None:
            metadata[f"LoRA {index}"] = f"{name} : {lora['weight']}{suffix}" if name else f"{lora['weight']}{suffix}"
        elif lora.get("weight_model") is not None or lora.get("weight_clip") is not None:
            metadata[f"LoRA {index}"] = (
                f"{name} : model={lora.get('weight_model')}, clip={lora.get('weight_clip')}{suffix}"
                if name else lora
            )
        else:
            metadata[f"LoRA {index}"] = f"{name}{suffix}" if name else lora

    return metadata


def _save_image_tensor(image):
    array = 255.0 * image.cpu().numpy()
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def _read_image_and_mask(image_path):
    img = node_helpers.pillow(Image.open, image_path)
    output_images = []
    output_masks = []
    width = None
    height = None

    for frame in ImageSequence.Iterator(img):
        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
        image = frame.convert("RGB")
        if width is None:
            width, height = image.size
        if image.size != (width, height):
            continue

        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
        if "A" in frame.getbands():
            mask = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        elif frame.mode == "P" and "transparency" in frame.info:
            mask = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask)
        else:
            mask_tensor = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        output_images.append(image_tensor)
        output_masks.append(mask_tensor.unsqueeze(0))
        if img.format == "MPO":
            break

    if len(output_images) > 1:
        return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
    return output_images[0], output_masks[0]


def _read_embedded_metadata(image_path):
    img = node_helpers.pillow(Image.open, image_path)
    embedded = {}
    for key, value in img.info.items():
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = value.hex()
        embedded[key] = _parse_json_maybe(value)

    exif = {}
    try:
        for key, value in img.getexif().items():
            exif[str(key)] = _parse_json_maybe(value)
    except Exception:
        pass

    return embedded, exif


def _read_sidecar_json(image_path):
    root, _ = os.path.splitext(image_path)
    sidecar_path = root + ".json"
    if not os.path.exists(sidecar_path):
        return None, sidecar_path

    with open(sidecar_path, "r", encoding="utf-8") as handle:
        return json.load(handle), sidecar_path


def _embedded_msv_metadata(embedded):
    metadata = embedded.get("metadata_saver_viewer")
    if isinstance(metadata, dict):
        return metadata
    return None


def _image_index_from_metadata(metadata):
    if not isinstance(metadata, dict):
        return None
    image = metadata.get("image")
    if not isinstance(image, dict):
        return None
    batch_index = image.get("batch_index")
    if batch_index is None:
        return None
    try:
        return int(batch_index)
    except (TypeError, ValueError):
        return None


class SaveImageWithMetadataJson:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "embed_png_metadata": ("BOOLEAN", {"default": True}),
                "save_sidecar_json": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata_json",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Saves images plus one JSON metadata file per image containing the full ComfyUI prompt/workflow."

    def save_images(self, images, filename_prefix="ComfyUI", embed_png_metadata=True, save_sidecar_json=True, prompt=None, extra_pnginfo=None):
        filename_prefix = _expand_filename_prefix(filename_prefix)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []
        metadata_files = []
        last_metadata = {}

        for batch_number, image in enumerate(images):
            img = _save_image_tensor(image)
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            image_file = f"{filename_with_batch_num}_{counter:05}_.png"
            image_path = os.path.join(full_output_folder, image_file)

            metadata = _build_metadata(prompt, extra_pnginfo, image_file, subfolder, batch_number)
            last_metadata = metadata

            pnginfo = None
            if embed_png_metadata and not args.disable_metadata:
                pnginfo = PngInfo()
                raw_metadata = metadata.get("raw", {})
                if raw_metadata.get("prompt") is not None:
                    pnginfo.add_text("prompt", json.dumps(raw_metadata["prompt"], default=_json_default))
                if raw_metadata.get("workflow") is not None:
                    pnginfo.add_text("workflow", json.dumps(raw_metadata["workflow"], default=_json_default))
                for key, value in raw_metadata.get("extra_pnginfo", {}).items():
                    pnginfo.add_text(key, json.dumps(value, default=_json_default))
                embedded_summary = {
                    key: value for key, value in metadata.items()
                    if key not in {"raw"}
                }
                pnginfo.add_text("metadata_saver_viewer", _pretty_json(embedded_summary))

            img.save(image_path, pnginfo=pnginfo, compress_level=self.compress_level)

            sidecar_file = None
            if save_sidecar_json:
                sidecar_file = os.path.splitext(image_file)[0] + ".json"
                sidecar_path = os.path.join(full_output_folder, sidecar_file)
                with open(sidecar_path, "w", encoding="utf-8") as handle:
                    handle.write(_pretty_json(metadata))
                    handle.write("\n")
                metadata_files.append({
                    "filename": sidecar_file,
                    "subfolder": subfolder,
                    "type": self.type,
                })

            results.append({
                "filename": image_file,
                "subfolder": subfolder,
                "type": self.type,
                "metadata": sidecar_file,
            })
            counter += 1

        return {
            "ui": {
                "images": results,
                "metadata_files": metadata_files,
                "text": [_pretty_json(last_metadata)],
            },
            "result": (_pretty_json(last_metadata),),
        }


class LoadImageMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "prefer_sidecar_json": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "metadata_json", "workflow_json")
    FUNCTION = "load_metadata"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads an image and shows embedded PNG/EXIF metadata plus matching sidecar JSON if it exists."

    def load_metadata(self, image, prefer_sidecar_json=True):
        image_path = folder_paths.get_annotated_filepath(image)
        image_tensor, mask_tensor = _read_image_and_mask(image_path)
        embedded, exif = _read_embedded_metadata(image_path)
        sidecar, sidecar_path = _read_sidecar_json(image_path)
        embedded_msv = _embedded_msv_metadata(embedded)
        embedded_image_index = _image_index_from_metadata(embedded_msv)

        raw_prompt = embedded.get("prompt")
        raw_extra = {key: value for key, value in embedded.items() if key not in {"prompt", "metadata_saver_viewer"}}

        metadata = {
            "image_path": image_path,
            "sidecar_json_path": sidecar_path if sidecar is not None else None,
            "sidecar_json": sidecar,
            "embedded": embedded,
            "exif": exif,
        }

        if sidecar is None and raw_prompt is not None:
            metadata["generated_summary"] = _build_metadata(
                raw_prompt,
                raw_extra,
                image_index=embedded_image_index,
            )

        if prefer_sidecar_json and sidecar is not None:
            display_metadata = sidecar
        elif embedded_msv is not None:
            display_metadata = embedded_msv
        elif "generated_summary" in metadata:
            display_metadata = metadata["generated_summary"]
        else:
            display_metadata = metadata

        workflow = None
        if isinstance(sidecar, dict):
            workflow = sidecar.get("raw", {}).get("workflow")
        if workflow is None:
            workflow = embedded.get("workflow")
        workflow_json = _pretty_json(workflow) if workflow is not None else ""
        metadata_json = _pretty_json(display_metadata)

        return {
            "ui": {"text": [metadata_json]},
            "result": (image_tensor, mask_tensor, metadata_json, workflow_json),
        }

    @classmethod
    def IS_CHANGED(cls, image, prefer_sidecar_json=True):
        image_path = folder_paths.get_annotated_filepath(image)
        hasher = hashlib.sha256()
        with open(image_path, "rb") as handle:
            hasher.update(handle.read())
        sidecar, sidecar_path = _read_sidecar_json(image_path)
        if sidecar is not None and os.path.exists(sidecar_path):
            with open(sidecar_path, "rb") as handle:
                hasher.update(handle.read())
        hasher.update(str(prefer_sidecar_json).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, prefer_sidecar_json=True):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "MSV_SaveImageWithMetadataJson": SaveImageWithMetadataJson,
    "MSV_LoadImageMetadata": LoadImageMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MSV_SaveImageWithMetadataJson": "Save Image + Metadata JSON",
    "MSV_LoadImageMetadata": "Load Image Metadata Viewer",
}
