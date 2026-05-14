# Metadata Saver Viewer for ComfyUI

Custom nodes:

- **Save Image + Metadata JSON** saves each image as PNG and writes a matching `.json` file next to it.
- **Load Image Metadata Viewer** uploads/loads an image and outputs the image, mask, metadata JSON text, and workflow JSON text.

The saved JSON uses a compact v2 layout and keeps:

- a Fooocus-like quick summary for prompts, sampler, model, LoRA, seed, steps, CFG, scheduler, and denoise when these can be inferred from the graph;
- the full raw ComfyUI `prompt`;
- the full raw ComfyUI `workflow`;
- any remaining raw `extra_pnginfo` fields.

`workflow` is stored once under `raw.workflow`, not duplicated inside `raw.extra_pnginfo`.

For `PerSampleLoraLoader`, the quick summary stores only the LoRA file name and the weight used for the saved image's batch index. Full LoRA settings remain in `raw.prompt`.

`filename_prefix` can include subfolders and date/time formatting, for example:

```text
%date:yyyy-MM-dd%/metedata_Test_
```

This saves files under `ComfyUI/output/2026-05-14/` with names like `metedata_Test_00001_.png` and `metedata_Test_00001_.json`.

For reliable workflow restore, keep `embed_png_metadata` enabled. ComfyUI can normally load an embedded workflow by dragging the generated PNG back onto the canvas.
