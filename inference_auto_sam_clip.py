from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from inference import run_inference


# ========== Monkey Patch：修复SAM的NMS设备不一致问题 ==========
def _apply_sam_nms_fix():
    """
    修复SAM的NMS（非极大值抑制）设备不一致问题
    强制在CPU上执行NMS，避免GPU/CPU设备冲突
    """
    try:
        import torchvision.ops.boxes as box_ops
        
        # 保存原始函数（可选，用于调试）
        original_batched_nms = box_ops.batched_nms
        
        def safe_batched_nms(boxes, scores, idxs, iou_threshold):
            """
            安全的NMS实现：
            1. 强制所有输入移到CPU
            2. 使用原生的_batched_nms_vanilla（稳定版本）
            3. 返回结果（保持与原始函数相同的返回类型）
            """
            # 确保输入在CPU上
            boxes_cpu = boxes.cpu() if boxes.is_cuda else boxes
            scores_cpu = scores.cpu() if scores.is_cuda else scores
            idxs_cpu = idxs.cpu() if idxs.is_cuda else idxs
            
            # 调用内部稳定版本
            keep = box_ops._batched_nms_vanilla(
                boxes_cpu, scores_cpu, idxs_cpu, iou_threshold
            )
            
            # 返回CPU上的索引（保持与原函数一致）
            return keep
        
        # 替换函数
        box_ops.batched_nms = safe_batched_nms
        print("[Monkey Patch] 已应用SAM NMS修复，强制CPU执行NMS")
        
    except ImportError as e:
        print(f"[Monkey Patch] 警告：无法应用NMS修复 - {e}")
    except Exception as e:
        print(f"[Monkey Patch] 警告：NMS修复失败 - {e}")


# 在模块加载时立即应用修复
_apply_sam_nms_fix()


DEFAULT_DATA_DIR = "/root/AssetDropper/data_auto_sam_clip"
DEFAULT_OUTPUT_DIR = "/root/AssetDropper/output_auto_sam_clip"
DEFAULT_PRETRAINED_MODEL = "/root/AssetDropper/models/AssetDropper"
DEFAULT_SAM_CKPT = "/root/AssetDropper/SAM/sam_vit_h_4b8939.pth"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


# ========== 辅助函数 ==========

def _image_to_base64(path: Path) -> str:
    """将图片转换为base64编码"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _glm_multimodal_once(
    image_path: Path,
    user_text: str,
    model: str,
) -> str:
    """调用智谱多模态模型"""
    from zai import ZhipuAiClient

    client = ZhipuAiClient(api_key=os.getenv("ZHIPUAI_API_KEY", ""))
    img_b64 = _image_to_base64(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_b64}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    }
    resp = client.chat.completions.create(**payload)
    return (resp.choices[0].message.content or "").strip()


def _glm_text_only(
    user_text: str,
    model: str,
) -> str:
    """纯文本调用智谱模型"""
    from zai import ZhipuAiClient

    client = ZhipuAiClient(api_key=os.getenv("ZHIPUAI_API_KEY", ""))
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": user_text,
            }
        ],
    }
    resp = client.chat.completions.create(**payload)
    return (resp.choices[0].message.content or "").strip()


def _sanitize_asset_prompt(text: str) -> str:
    """清理AssetDropper的prompt，移除禁止词汇"""
    lowered = text.lower()
    banned = [
        "text", "letters", "numbers", "numeric",
        "word", "words", "typography", "font", "ocr",
    ]
    out = lowered
    for w in banned:
        out = out.replace(w, "")
    out = " ".join(out.replace(" ,", ",").replace(",,", ",").split())
    return out.strip(" ,.")


def generate_asset_prompt_direct(
    user_hint: str,  # 可以是中文或英文
    image_path: Path,
    zhipu_model: str,
) -> str:
    """
    直接生成AssetDropper的资产描述prompt
    支持中文输入，一次调用完成
    """
    user_prompt = f"""Generate one English prompt for AssetDropper.
Target asset: {user_hint}

Rules:
1) Describe ONLY the extractable asset/object in the image.
2) Keep key visual semantics: shape, color palette, parts, material/texture.
3) Use a concise comma-separated prompt.
4) The object must be complete and centered.
5) Avoid words related to OCR/text content.
6) Add "clean background, high detail".
7) Output ONE line only.
"""
    try:
        result = _glm_multimodal_once(
            image_path=image_path,
            user_text=user_prompt,
            model=zhipu_model,
        )
        result = _sanitize_asset_prompt(result.replace("\n", " "))
        if not result:
            result = f"{user_hint}, illustration, graphic, clean background, high detail"
        return result
    except Exception as e:
        print(f"[warn] Asset prompt failed: {e}")
        return f"{user_hint}, illustration, graphic, clean background, high detail"


def _extract_clip_prompt_from_asset_prompt(asset_prompt: str) -> str:
    """
    从AssetDropper prompt中提取用于CLIP的简洁prompt
    """
    generic_suffixes = [
        "clean background", "high detail", "illustration", "graphic",
        "centered", "complete object", "full object"
    ]
    
    parts = [p.strip() for p in asset_prompt.split(",")]
    
    filtered_parts = []
    for part in parts:
        part_lower = part.lower()
        is_generic = any(suffix.lower() in part_lower for suffix in generic_suffixes)
        if not is_generic:
            filtered_parts.append(part)
    
    if not filtered_parts:
        filtered_parts = [parts[0]] if parts else [asset_prompt]
    
    clip_prompt = filtered_parts[0]
    
    if len(filtered_parts) > 1 and len(filtered_parts[1]) < 20:
        clip_prompt = f"{filtered_parts[0]}, {filtered_parts[1]}"
    
    return clip_prompt


def _clip_select_mask_with_asset_prompt(
    image_rgb: np.ndarray,
    masks: list[dict],
    asset_prompt: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: str,
    min_area_ratio: float,
    max_area_ratio: float,
    use_enhanced_for_clip: bool = False,
) -> np.ndarray:
    """使用AssetDropper的prompt作为CLIP的语义参考"""
    if use_enhanced_for_clip:
        clip_prompt = asset_prompt
        print(f"[CLIP] 使用完整AssetDropper prompt: '{clip_prompt[:80]}...'")
    else:
        clip_prompt = _extract_clip_prompt_from_asset_prompt(asset_prompt)
        print(f"[CLIP] 提取核心语义: '{clip_prompt}'")
    
    best_mask = None
    best_score = -1e9
    
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        seg = np.asarray(seg, dtype=bool)
        if seg.ndim != 2:
            seg = np.squeeze(seg)
        area_ratio = float(seg.mean())
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        ys, xs = np.where(seg)
        if ys.size == 0 or xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = image_rgb[y0:y1+1, x0:x1+1].copy()
        crop_mask = seg[y0:y1+1, x0:x1+1]
        crop[~crop_mask] = 127

        inputs = clip_processor(
            text=[clip_prompt],
            images=[crop],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_feat = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_feat = clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            score = float((image_feat * text_feat).sum(dim=-1).item())

        area_bonus = 0.05 if area_ratio > 0.01 else 0.0
        final_score = score + area_bonus
        
        print(f"  Mask area: {area_ratio:.3f}, Score: {score:.4f}, Final: {final_score:.4f}")
        
        if final_score > best_score:
            best_score = final_score
            best_mask = seg

    if best_mask is None:
        largest = max(masks, key=lambda x: int(x.get("area", 0)))
        best_mask = np.asarray(largest["segmentation"], dtype=bool)
        print(f"[warn] 使用fallback: 最大面积mask")
    
    return best_mask


def _ensure_dirs(data_dir: Path) -> None:
    """确保数据目录结构存在"""
    (data_dir / "Image").mkdir(parents=True, exist_ok=True)
    (data_dir / "Mask").mkdir(parents=True, exist_ok=True)
    (data_dir / "Caption").mkdir(parents=True, exist_ok=True)
    (data_dir / "Prompt").mkdir(parents=True, exist_ok=True)


def _write_single_line(path: Path, text: str) -> None:
    """写入单行文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def _write_lines(path: Path, lines: List[str]) -> None:
    """写入多行文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


def _collect_input_images(args: argparse.Namespace) -> List[Path]:
    """收集输入图像"""
    if bool(args.input_image) == bool(args.input_dir):
        raise ValueError("必须二选一：仅传 --input_image 或仅传 --input_dir")
    if args.input_image:
        p = Path(args.input_image)
        if not p.exists():
            raise FileNotFoundError(f"--input_image not found: {p}")
        return [p]
    pdir = Path(args.input_dir)
    if not pdir.exists() or not pdir.is_dir():
        raise FileNotFoundError(f"--input_dir not found or not directory: {pdir}")
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = [p for p in sorted(pdir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not images:
        raise RuntimeError(f"no image files found in --input_dir: {pdir}")
    return images


def _save_image_512(src: Path, dst: Path) -> np.ndarray:
    """保存512x512图像"""
    img = Image.open(src).convert("RGB").resize((512, 512))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return np.array(img, dtype=np.uint8)


def _build_sam_generator(
    checkpoint: str,
    model_type: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    crop_n_layers: int,
    crop_n_points_downscale_factor: int,
):
    """构建SAM生成器（改进版：支持小目标检测）"""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    
    return sam, SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="SAM grid mask pool + CLIP rerank + AssetDropper")
    
    # 输入输出
    p.add_argument("--input_image", type=str, default=None)
    p.add_argument("--input_dir", type=str, default=None)
    p.add_argument("--target", type=str, required=True, 
                   help="目标资产描述（支持中文）")
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--list_name", type=str, default="auto_list")
    
    # ========== SAM 精细化配置 ==========
    p.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM_CKPT)
    p.add_argument("--sam_model_type", type=str, default="vit_h", 
                   choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--sam_device", type=str, default="cuda")
    p.add_argument("--sam_fallback_cpu_on_nms_error", action="store_true", default=True,
                   help="当SAM在GPU出现NMS设备不一致错误时，自动回退到CPU重试")
    p.add_argument("--sam_points_per_side", type=int, default=48,
                   help="网格点数（越高越能检测小目标，推荐48-64）")
    p.add_argument("--sam_pred_iou_thresh", type=float, default=0.80,
                   help="预测IoU阈值（降低可获得更多候选mask）")
    p.add_argument("--sam_stability_score_thresh", type=float, default=0.85,
                   help="稳定性分数阈值（降低可获得更多候选mask）")
    p.add_argument("--sam_crop_n_layers", type=int, default=2,
                   help="裁剪层数（增加可改善小目标检测）")
    p.add_argument("--sam_crop_n_points_downscale_factor", type=int, default=1,
                   help="裁剪点下采样因子（1表示精细）")
    
    # CLIP 配置
    p.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL)
    p.add_argument("--clip_device", type=str, default="cuda")
    p.add_argument("--min_mask_area_ratio", type=float, default=0.005,
                   help="最小mask面积比例")
    p.add_argument("--max_mask_area_ratio", type=float, default=0.9,
                   help="最大mask面积比例")
    p.add_argument("--clip_use_full_asset_prompt", action="store_true",
                   help="CLIP使用完整的AssetDropper prompt（否则提取核心语义）")
    
    # 大模型配置
    p.add_argument("--asset_zhipu_model", type=str, default="glm-4.6v",
                   help="用于生成prompt的智谱模型")
    
    # AssetDropper 推理配置
    p.add_argument("--pretrained_model_name_or_path", type=str, default=DEFAULT_PRETRAINED_MODEL)
    p.add_argument("--num_inference_steps", type=int, default=100,
                   help="去噪步数（推荐50-100）")
    p.add_argument("--test_batch_size", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=2.5,
                   help="CFG引导强度（推荐2.5-3.0）")
    p.add_argument("--mixed_precision", type=str, default="fp16", 
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_images = _collect_input_images(args)
    data_dir = Path(args.data_dir)
    _ensure_dirs(data_dir)
    
    # ========== 加载模型 ==========
    print("[加载] CLIP模型...")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(args.clip_device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    
    print("[加载] SAM模型...")
    sam, mask_generator = _build_sam_generator(
        checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device,
        points_per_side=args.sam_points_per_side,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        crop_n_layers=args.sam_crop_n_layers,
        crop_n_points_downscale_factor=args.sam_crop_n_points_downscale_factor,
    )
    
    # ========== 处理每张图像 ==========
    list_names: List[str] = []
    for i, src in enumerate(input_images, start=1):
        name = src.stem
        img_dst = data_dir / "Image" / f"{name}.png"
        mask_dst = data_dir / "Mask" / f"{name}.png"
        cap_dst = data_dir / "Caption" / f"{name}.txt"
        prompt_dst = data_dir / "Prompt" / f"{name}_asset.txt"
        
        print(f"\n[处理 {i}/{len(input_images)}] {name}")
        
        # 1. 保存512x512图像
        print(f"  [1/4] 图像预处理...")
        image_rgb = _save_image_512(src, img_dst)
        
        # 2. 直接生成AssetDropper prompt（一次调用，支持中文输入）
        print(f"  [2/4] 生成AssetDropper prompt...")
        asset_prompt = generate_asset_prompt_direct(
            user_hint=args.target,  # 直接使用原始输入（支持中文）
            image_path=img_dst,
            zhipu_model=args.asset_zhipu_model,
        )
        _write_single_line(prompt_dst, asset_prompt)
        print(f"       Asset prompt: {asset_prompt[:80]}...")
        
        # 3. SAM生成候选mask
        print(f"  [3/4] SAM生成候选mask (points_per_side={args.sam_points_per_side}, crop_n_layers={args.sam_crop_n_layers})...")
        try:
            masks = mask_generator.generate(image_rgb)
        except RuntimeError as e:
            emsg = str(e)
            nms_dev_err = "indices should be either on cpu or on the same device as the indexed tensor" in emsg
            if nms_dev_err and args.sam_fallback_cpu_on_nms_error:
                print("  [warn] SAM GPU NMS设备不一致，自动回退CPU重试...")
                try:
                    del mask_generator
                    del sam
                except Exception:
                    pass
                if args.sam_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                sam, mask_generator = _build_sam_generator(
                    checkpoint=args.sam_checkpoint,
                    model_type=args.sam_model_type,
                    device="cpu",
                    points_per_side=args.sam_points_per_side,
                    pred_iou_thresh=args.sam_pred_iou_thresh,
                    stability_score_thresh=args.sam_stability_score_thresh,
                    crop_n_layers=args.sam_crop_n_layers,
                    crop_n_points_downscale_factor=args.sam_crop_n_points_downscale_factor,
                )
                masks = mask_generator.generate(image_rgb)
            else:
                raise
        if not masks:
            print(f"  [warn] SAM无候选，跳过")
            continue
        print(f"       生成了 {len(masks)} 个候选mask")
        
        # 4. CLIP选择最佳mask（使用AssetDropper的prompt）
        print(f"  [4/4] CLIP选择最佳mask...")
        best_mask = _clip_select_mask_with_asset_prompt(
            image_rgb=image_rgb,
            masks=masks,
            asset_prompt=asset_prompt,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=args.clip_device,
            min_area_ratio=args.min_mask_area_ratio,
            max_area_ratio=args.max_mask_area_ratio,
            use_enhanced_for_clip=args.clip_use_full_asset_prompt,
        )
        mask_u8 = (best_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(mask_dst), mask_u8)
        print(f"       mask保存至: {mask_dst}")
        
        # 保存caption
        _write_single_line(cap_dst, asset_prompt)
        
        list_names.append(img_dst.name)
    
    if not list_names:
        raise RuntimeError("没有可用样本，流程结束")
    _write_lines(data_dir / f"{args.list_name}.txt", list_names)
    
    # ========== 释放显存 ==========
    print("\n[清理] 释放SAM/CLIP显存...")
    try:
        del mask_generator
        del sam
        del clip_model
        del clip_processor
    except Exception:
        pass
    if args.sam_device.startswith("cuda") or args.clip_device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    # ========== 运行AssetDropper推理 ==========
    print("\n[推理] 启动AssetDropper...")
    infer_args = argparse.Namespace(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        width=512,
        height=512,
        Pwidth=512,
        Pheight=512,
        txt_name=args.list_name,
        num_inference_steps=args.num_inference_steps,
        output_dir=args.output_dir,
        data_dir=str(data_dir),
        seed=args.seed,
        test_batch_size=args.test_batch_size,
        guidance_scale=args.guidance_scale,
        mixed_precision=args.mixed_precision,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        run_inference(infer_args)
    except torch.OutOfMemoryError:
        print("[warn] OOM detected, retry with test_batch_size=1 ...")
        torch.cuda.empty_cache()
        infer_args.test_batch_size = 1
        run_inference(infer_args)
    
    print(f"\n[完成] 所有处理完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()