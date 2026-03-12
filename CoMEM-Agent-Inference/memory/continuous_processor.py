"""Shared helpers for preparing continuous-memory trajectory tensors."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Iterable, Sequence

from PIL import Image

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"


def decode_base64_image(image_base64: str | dict | None):
    """Decode a base64-encoded image payload into a PIL image."""
    if not image_base64:
        return None
    if isinstance(image_base64, dict):
        image_base64 = image_base64.get("url")
    if not image_base64:
        return None
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes))


def serialize_action(action: object) -> str:
    """Serialize a stored action dict into a compact text payload."""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        name = action.get("name") or action.get("action") or action.get("action_type")
        arguments = action.get("arguments", {})
        if isinstance(arguments, dict):
            reasoning = arguments.get("reasoning", "")
            description = arguments.get("description", "")
            payload = {"name": name, "reasoning": reasoning, "description": description}
            return json.dumps(payload, ensure_ascii=True)
    return json.dumps(action, ensure_ascii=True)


def build_experience_inputs(processor, texts=None, images=None):
    """Prepare processor inputs for continuous memory."""
    texts = texts or []
    images = images or []

    all_experience_input_ids = []
    all_experience_pixel_values = []
    all_experience_image_grid_thw = []

    for trajectory_actions, trajectory_images in zip(texts, images):
        trajectory_text = ""
        trajectory_image = []

        for action, image_base64 in zip(trajectory_actions, trajectory_images):
            serialized_action = serialize_action(action)
            image = decode_base64_image(image_base64)
            if image is not None:
                trajectory_image.append(image)
                trajectory_text += (
                    f"{DEFAULT_IM_START_TOKEN}user\n"
                    f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}"
                    f"{serialized_action}{DEFAULT_IM_END_TOKEN}\n"
                )
            else:
                trajectory_text += (
                    f"{DEFAULT_IM_START_TOKEN}user\n"
                    f"{serialized_action}{DEFAULT_IM_END_TOKEN}\n"
                )

        if trajectory_image:
            e_inputs = processor(
                text=[trajectory_text],
                images=trajectory_image,
                padding=False,
                return_tensors="pt",
            )
            e_input_ids = e_inputs["input_ids"].squeeze(0)
            all_experience_pixel_values.append(e_inputs["pixel_values"])
            all_experience_image_grid_thw.append(e_inputs["image_grid_thw"])
        else:
            e_input_ids = processor.tokenizer(
                trajectory_text,
                add_special_tokens=False,
                padding=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

        all_experience_input_ids.append(e_input_ids)

    return {
        "experience_input_ids": all_experience_input_ids,
        "experience_pixel_values": all_experience_pixel_values,
        "experience_image_grid_thw": all_experience_image_grid_thw,
    }


def attach_experience_inputs(processor, inputs, texts=None, images=None):
    """Attach prepared continuous-memory tensors to an existing model input dict."""
    inputs.update(build_experience_inputs(processor=processor, texts=texts, images=images))
    return inputs

