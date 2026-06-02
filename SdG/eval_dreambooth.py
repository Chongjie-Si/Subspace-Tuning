#!/usr/bin/env python3
"""
DreamBooth quantitative evaluation script for MAP paper.

Computes:
  - CLIP-T  : text-image alignment (generated image vs. prompt)
  - CLIP-I  : image-image similarity (generated image vs. reference subject images)
  - DINO    : image-image similarity using DINO ViT features
  - LPIPS   : perceptual diversity among generated images

Usage:
    python eval_dreambooth.py \
        --model_path ./lora-trained-xl-cat \
        --subject cat \
        --instance_dir ./cat \
        --base_model stabilityai/stable-diffusion-xl-base-1.0 \
        --num_images 25 \
        --seed 42

Requirements:
    pip install torch transformers diffusers accelerate pillow lpips open_clip_torch
    pip install git+https://github.com/facebookresearch/dinov2  # or use torchvision DINO
"""

import argparse
import os
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


SUBJECT_PROMPTS = [
    "A picture of a {} in the jungle",
    "A picture of a {} in the snow",
    "A picture of a {} on the beach",
    "A picture of a {} on a cobblestone street",
    "A picture of a {} on top of pink fabric",
    "A picture of a {} on top of a wooden floor",
    "A picture of a {} with a city in the background",
    "A picture of a {} with a mountain in the background",
    "A picture of a {} with a blue house in the background",
    "A picture of a {} on top of a purple rug in a forest",
    "A picture of a {} with a wheat field in the background",
    "A picture of a {} with a tree and autumn leaves in the background",
    "A picture of a {} with the Eiffel Tower in the background",
    "A picture of a {} floating on top of water",
    "A picture of a {} floating in an ocean of milk",
    "A picture of a {} on top of green grass with sunflowers around it",
    "A picture of a {} on top of a mirror",
    "A picture of a {} on top of the sidewalk in a crowded street",
    "A picture of a {} on top of a dirt road",
    "A picture of a {} on top of a white rug",
    "A picture of a red {}",
    "A picture of a purple {}",
    "A picture of a shiny {}",
    "A picture of a wet {}",
    "A picture of a cube shaped {}",
]


def generate_images(model_path, base_model, subject, prompts, output_dir, num_inference_steps=50, seed=42):
    from diffusers import DiffusionPipeline

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(model_path)

    generator = torch.Generator("cuda").manual_seed(seed)
    images = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        image = pipe(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
        out_path = os.path.join(output_dir, f"img_{i:03d}.jpg")
        image.save(out_path)
        images.append(image)
    return images


def load_reference_images(instance_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in Path(instance_dir).iterdir() if p.suffix.lower() in exts]
    return [Image.open(p).convert("RGB") for p in sorted(paths)]


def compute_clip_t(images, prompts, device="cuda"):
    """CLIP text-image alignment score."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    scores = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            text_tokens = tokenizer([prompt]).to(device)
            img_feat = model.encode_image(img_tensor)
            txt_feat = model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores.append((img_feat * txt_feat).sum().item())
    return float(np.mean(scores))


def compute_clip_i(gen_images, ref_images, device="cuda"):
    """CLIP image-image similarity between generated and reference images."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    with torch.no_grad():
        ref_feats = []
        for img in ref_images:
            t = preprocess(img).unsqueeze(0).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            ref_feats.append(f)
        ref_feat = torch.cat(ref_feats, dim=0).mean(0, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)

        scores = []
        for img in gen_images:
            t = preprocess(img).unsqueeze(0).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            scores.append((f * ref_feat).sum().item())
    return float(np.mean(scores))


def compute_dino_i(gen_images, ref_images, device="cuda"):
    """DINO image-image similarity between generated and reference images."""
    import torchvision.transforms as T
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model = model.to(device).eval()

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract(imgs):
        feats = []
        with torch.no_grad():
            for img in imgs:
                t = transform(img).unsqueeze(0).to(device)
                f = model(t)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f)
        return torch.cat(feats, dim=0)

    ref_feats = extract(ref_images).mean(0, keepdim=True)
    ref_feats = ref_feats / ref_feats.norm(dim=-1, keepdim=True)
    gen_feats = extract(gen_images)
    scores = (gen_feats * ref_feats).sum(-1).cpu().numpy()
    return float(np.mean(scores))


def compute_lpips_diversity(images, device="cuda"):
    """Average pairwise LPIPS distance among generated images (diversity)."""
    import lpips
    loss_fn = lpips.LPIPS(net="vgg").to(device)
    import torchvision.transforms.functional as TF

    def to_tensor(img):
        t = TF.to_tensor(img.resize((256, 256))).unsqueeze(0).to(device)
        return t * 2 - 1  # normalize to [-1,1]

    tensors = [to_tensor(img) for img in images]
    dists = []
    with torch.no_grad():
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                d = loss_fn(tensors[i], tensors[j]).item()
                dists.append(d)
    return float(np.mean(dists)) if dists else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the trained LoRA weights directory")
    parser.add_argument("--subject", required=True, help="Subject name (e.g., 'cat')")
    parser.add_argument("--instance_dir", required=True, help="Directory with reference subject images")
    parser.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", default=None, help="Where to save generated images (default: <model_path>/eval_images)")
    parser.add_argument("--results_file", default=None, help="JSON file to save results (default: <model_path>/eval_results.json)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation, load from output_dir")
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS diversity computation (slow)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "eval_images")
    if args.results_file is None:
        args.results_file = os.path.join(args.model_path, "eval_results.json")

    prompts = [p.format(args.subject) for p in SUBJECT_PROMPTS]

    # Generate or load images
    if args.skip_generation:
        exts = {".jpg", ".jpeg", ".png"}
        gen_images = [Image.open(p).convert("RGB")
                      for p in sorted(Path(args.output_dir).iterdir())
                      if p.suffix.lower() in exts]
        print(f"Loaded {len(gen_images)} generated images from {args.output_dir}")
    else:
        print("Generating images...")
        gen_images = generate_images(
            args.model_path, args.base_model, args.subject, prompts,
            args.output_dir, args.num_inference_steps, args.seed
        )

    # Load reference images
    ref_images = load_reference_images(args.instance_dir)
    print(f"Loaded {len(ref_images)} reference images from {args.instance_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    print("Computing CLIP-T (text alignment)...")
    results["clip_t"] = compute_clip_t(gen_images[:len(prompts)], prompts[:len(gen_images)], device=device)
    print(f"  CLIP-T: {results['clip_t']:.4f}")

    print("Computing CLIP-I (subject fidelity)...")
    results["clip_i"] = compute_clip_i(gen_images, ref_images, device=device)
    print(f"  CLIP-I: {results['clip_i']:.4f}")

    print("Computing DINO (subject fidelity)...")
    results["dino"] = compute_dino_i(gen_images, ref_images, device=device)
    print(f"  DINO:   {results['dino']:.4f}")

    if not args.no_lpips:
        print("Computing LPIPS diversity...")
        results["lpips_diversity"] = compute_lpips_diversity(gen_images, device=device)
        print(f"  LPIPS diversity: {results['lpips_diversity']:.4f}")

    results["subject"] = args.subject
    results["model_path"] = args.model_path
    results["seed"] = args.seed
    results["num_images"] = len(gen_images)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
