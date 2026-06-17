"""
AprilTag marker detection (tag36h11 family) — pupil_apriltags:
  multi-pass (CLAHE / histogram), scaling for small frames,
  fallback for stickers without white border (black square crop + pad),
  ID 0–4 filter, deduplication and geometric layout validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pupil_apriltags import Detector

from config import AprilTagConfig


@dataclass
class FiducialDetection:
    corners: list  # list of (1,4,2) float32 — order matches OpenCV solvePnP
    ids: np.ndarray  # (N,1) int32
    method: str  # apriltag | apriltag_crop | none


def _corners_to_opencv_order(corners: np.ndarray) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return c.reshape(1, 4, 2)


def _highlight_suppressed(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Flatten specular hotspots by dividing out a large-scale brightness map."""
    sig = max(3.0, float(sigma))
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sig, sigmaY=sig)
    blur = np.maximum(blur.astype(np.float32), 8.0)
    norm = cv2.divide(gray.astype(np.float32), blur, scale=128.0)
    return np.clip(norm, 0, 255).astype(np.uint8)


def _gamma_correct(gray: np.ndarray, gamma: float) -> np.ndarray:
    g = max(0.2, float(gamma))
    inv = 1.0 / g
    table = (np.arange(256, dtype=np.float32) / 255.0) ** inv * 255.0
    return cv2.LUT(gray, np.clip(table, 0, 255).astype(np.uint8))


def _unsharp_mask(
    gray: np.ndarray,
    sigma: float,
    amount: float,
    threshold: int,
) -> np.ndarray:
    sig = max(0.3, float(sigma))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sig, sigmaY=sig)
    sharp = cv2.addWeighted(gray, 1.0 + float(amount), blurred, -float(amount), 0)
    if threshold > 0:
        low_contrast = np.abs(gray.astype(np.int16) - blurred.astype(np.int16)) < int(threshold)
        out = sharp.copy()
        out[low_contrast] = gray[low_contrast]
        return out
    return sharp


def _adaptive_binary(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    block = max(11, int(block_size) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        int(c),
    )


def _prepare_gray(gray: np.ndarray, tag_cfg: AprilTagConfig) -> np.ndarray:
    """One-time cleanup per frame: denoise MJPEG blocks, then unsharp soft focus."""
    out = gray
    if tag_cfg.use_denoise:
        d = max(3, int(tag_cfg.denoise_d) | 1)
        out = cv2.bilateralFilter(
            out,
            d,
            float(tag_cfg.denoise_sigma_color),
            float(tag_cfg.denoise_sigma_space),
        )
    if tag_cfg.use_unsharp_mask:
        out = _unsharp_mask(
            out,
            float(tag_cfg.unsharp_sigma),
            float(tag_cfg.unsharp_amount),
            int(tag_cfg.unsharp_threshold),
        )
    return out


def _clahe(tag_cfg: AprilTagConfig) -> cv2.CLAHE:
    clip = max(1.0, float(tag_cfg.clahe_clip_limit))
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))


def _gray_variants(gray: np.ndarray, tag_cfg: AprilTagConfig) -> list[tuple[str, np.ndarray]]:
    """Ordered preprocessing passes — cheap ones first, glare-specific before heavy equalize."""
    out: list[tuple[str, np.ndarray]] = [("prep", gray)]
    if tag_cfg.use_highlight_suppression:
        out.append(
            (
                "glare",
                _highlight_suppressed(gray, float(tag_cfg.highlight_blur_sigma)),
            )
        )
    if tag_cfg.use_clahe:
        clahe = _clahe(tag_cfg)
        out.append(("clahe", clahe.apply(gray)))
        if tag_cfg.use_highlight_suppression:
            out.append(
                (
                    "clahe_glare",
                    clahe.apply(
                        _highlight_suppressed(gray, float(tag_cfg.highlight_blur_sigma))
                    ),
                )
            )
    if tag_cfg.use_adaptive_threshold:
        out.append(
            (
                "adapt",
                _adaptive_binary(
                    gray,
                    int(tag_cfg.adaptive_block_size),
                    int(tag_cfg.adaptive_c),
                ),
            )
        )
    if tag_cfg.use_histogram_eq:
        out.append(("histeq", cv2.equalizeHist(gray)))
    out.append(("gamma_dark", _gamma_correct(gray, 0.65)))
    return out


def _extract_black_tag_patch(
    gray: np.ndarray,
    thresh: int,
    pad_frac: float,
    *,
    max_area_frac: float = 0.35,
    min_area_frac: float = 0.002,
) -> tuple[np.ndarray, int, int, int, int, int, int] | None:
    """
    Extract dark tag square (without red backing) and add white border.
    Returns: patch, x0, y0, bw, bh, pad_px, upscale_factor.
    """
    _, black = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, k)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    img_area = float(gray.shape[0] * gray.shape[1])
    max_area = img_area * max_area_frac
    min_area = max(900.0, img_area * min_area_frac)
    best_score = 0.0
    best_rect: tuple[int, int, int, int] | None = None

    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        x0, y0, bw, bh = cv2.boundingRect(c)
        if bw < 20 or bh < 20:
            continue
        aspect = bw / float(bh)
        if aspect < 0.55 or aspect > 1.8:
            continue
        fill = area / float(bw * bh)
        if fill < 0.35:
            continue
        score = area * min(aspect, 1.0 / aspect) * fill
        if score > best_score:
            best_score = score
            best_rect = (x0, y0, bw, bh)

    if best_rect is None:
        return None

    x0, y0, bw, bh = best_rect
    pad_px = max(12, int(max(bw, bh) * pad_frac))
    roi = gray[y0 : y0 + bh, x0 : x0 + bw]
    patch = cv2.copyMakeBorder(roi, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=255)
    return patch, x0, y0, bw, bh, pad_px, 2


def _remap_corners_from_patch(
    corners: np.ndarray,
    x0: int,
    y0: int,
    pad_px: int,
    upscale: int,
) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    inv = 1.0 / float(upscale)
    c = c * inv - float(pad_px)
    c[:, 0] += float(x0)
    c[:, 1] += float(y0)
    return c.reshape(1, 4, 2)


class FiducialDetector:
    """AprilTag detection backend per frame."""

    def __init__(self, tag_cfg: AprilTagConfig) -> None:
        self._tag = tag_cfg
        self._allowed = sorted(int(x) for x in tag_cfg.marker_ids)
        self._allowed_set = set(self._allowed)
        self._detector = Detector(
            families=tag_cfg.family,
            nthreads=2,
            quad_decimate=float(tag_cfg.quad_decimate),
            quad_sigma=float(tag_cfg.quad_sigma),
            refine_edges=1 if tag_cfg.refine_edges else 0,
            decode_sharpening=float(tag_cfg.decode_sharpening),
        )
        # ROI tracking state (crop around last detection).
        self._last_roi: tuple[int, int, int, int] | None = None

    def _filter_allowed(
        self, corners: list | None, ids: np.ndarray | None
    ) -> tuple[list, np.ndarray]:
        if corners is None or ids is None or len(ids) == 0:
            return [], np.empty((0, 1), dtype=np.int32)
        kept_c: list = []
        kept_ids: list[int] = []
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if mid in self._allowed_set:
                kept_c.append(corners[i])
                kept_ids.append(mid)
        if not kept_ids:
            return [], np.empty((0, 1), dtype=np.int32)
        return kept_c, np.array([[x] for x in kept_ids], dtype=np.int32)

    def _dedupe_by_id(self, corners: list, ids: np.ndarray) -> tuple[list, np.ndarray]:
        best: dict[int, tuple[np.ndarray, float]] = {}
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            c = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
            area = float(cv2.contourArea(c))
            if mid not in best or area > best[mid][1]:
                best[mid] = (corners[i], area)
        cap = max(1, int(self._tag.max_markers_per_frame))
        ranked = sorted(best.items(), key=lambda x: x[1][1], reverse=True)[:cap]
        out_c = [x[1][0] for x in ranked]
        out_ids = np.array([[x[0]] for x in ranked], dtype=np.int32)
        return out_c, out_ids

    def _validate_layout(self, corners: list, ids: np.ndarray) -> tuple[list, np.ndarray]:
        if len(ids) < 2:
            return corners, ids
        centers: dict[int, np.ndarray] = {}
        areas: dict[int, float] = {}
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            c = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
            centers[mid] = c.mean(axis=0)
            areas[mid] = float(cv2.contourArea(c))
        med_area = float(np.median(list(areas.values())))
        keep_ids: list[int] = []
        keep_c: list = []
        for i, mid in enumerate(ids.flatten()):
            mid = int(mid)
            if med_area > 0 and areas[mid] < med_area * 0.08:
                continue
            if med_area > 0 and areas[mid] > med_area * 10.0:
                continue
            keep_ids.append(mid)
            keep_c.append(corners[i])
        if not keep_ids:
            return corners, ids
        return keep_c, np.array([[x] for x in keep_ids], dtype=np.int32)

    def _collect_detections(
        self,
        gray: np.ndarray,
        margin_thr: float,
        remap: tuple[int, int, int, int] | None = None,
    ) -> tuple[list, list[int]]:
        """Collect hits at current image scale (optionally patch -> full frame)."""
        corners: list = []
        ids_list: list[int] = []
        detections = self._detector.detect(gray)
        for det in detections:
            if det.decision_margin is not None and float(det.decision_margin) < margin_thr:
                continue
            mid = int(det.tag_id)
            if mid not in self._allowed_set:
                continue
            c = _corners_to_opencv_order(det.corners)
            if remap is not None:
                x0, y0, pad_px, upscale = remap
                c = _remap_corners_from_patch(c, x0, y0, pad_px, upscale)
            corners.append(c)
            ids_list.append(mid)
        return corners, ids_list

    def _single_pass(self, gray: np.ndarray, margin_thr: float) -> tuple[list, list[int], str]:
        """Try multiple gray variants until a tag is found."""
        for name, variant in _gray_variants(gray, self._tag):
            corners, ids_list = self._collect_detections(variant, margin_thr)
            if ids_list:
                return corners, ids_list, name
        return [], [], "none"

    def _detect_apriltag(self, gray: np.ndarray) -> tuple[list, np.ndarray, str]:
        """Multi-pass detect + optional upscale + black-square crop fallback."""
        h, w = gray.shape[:2]
        margin_thr = float(self._tag.min_decision_margin)

        corners, ids_list, variant = self._single_pass(gray, margin_thr)
        if ids_list:
            method = "apriltag" if variant == "prep" else f"apriltag_{variant}"
            return corners, np.array([[x] for x in ids_list], dtype=np.int32), method

        upscale_scales = (
            [s for s in self._tag.detection_upscales if float(s) > 1.0]
            if self._tag.detection_lazy_upscale
            else []
        )
        for scale in upscale_scales:
                inv = 1.0 / float(scale)
                g = cv2.resize(
                    gray,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
                corners, ids_list, variant = self._single_pass(g, margin_thr)
                if ids_list:
                    corners = [np.asarray(c, dtype=np.float32) * inv for c in corners]
                    method = "apriltag_up" if variant == "prep" else f"apriltag_up_{variant}"
                    return corners, np.array([[x] for x in ids_list], dtype=np.int32), method

        if self._tag.use_black_crop_fallback:
            crop_corners, crop_ids = self._black_crop_fallback(gray, margin_thr)
            if len(crop_ids):
                return crop_corners, crop_ids, "apriltag_crop"

        return [], np.empty((0, 1), dtype=np.int32), "none"

    def _black_crop_fallback(
        self, gray: np.ndarray, margin_thr: float
    ) -> tuple[list, np.ndarray]:
        upscale = max(1, int(self._tag.black_crop_upscale))
        for thr in self._tag.black_crop_thresholds:
            patch_info = _extract_black_tag_patch(
                gray,
                int(thr),
                float(self._tag.black_crop_pad_frac),
            )
            if patch_info is None:
                continue
            patch, x0, y0, _bw, _bh, pad_px, _ = patch_info
            if upscale > 1:
                patch = cv2.resize(
                    patch,
                    (patch.shape[1] * upscale, patch.shape[0] * upscale),
                    interpolation=cv2.INTER_CUBIC,
                )
            corners, ids_list = self._collect_detections(
                patch,
                margin_thr,
                remap=(x0, y0, pad_px, upscale),
            )
            if ids_list:
                return corners, np.array([[x] for x in ids_list], dtype=np.int32)
        return [], np.empty((0, 1), dtype=np.int32)

    def _norm_roi_pixels(
        self, w: int, h: int
    ) -> tuple[int, int, int, int] | None:
        roi = self._tag.search_roi_norm
        if not roi or len(roi) != 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in roi)
        rx0 = int(np.clip(min(x0, x1), 0.0, 1.0) * w)
        ry0 = int(np.clip(min(y0, y1), 0.0, 1.0) * h)
        rx1 = int(np.clip(max(x0, x1), 0.0, 1.0) * w)
        ry1 = int(np.clip(max(y0, y1), 0.0, 1.0) * h)
        if rx1 - rx0 < 32 or ry1 - ry0 < 32:
            return None
        return rx0, ry0, rx1, ry1

    def _map_patch_corners(
        self,
        corners: list,
        offset_x: float,
        offset_y: float,
        patch_scale: float,
    ) -> list:
        if patch_scale == 1.0:
            shift = np.array([offset_x, offset_y], dtype=np.float32)
            return [
                np.asarray(c, dtype=np.float32).reshape(1, 4, 2) + shift for c in corners
            ]
        inv = 1.0 / float(patch_scale)
        shift = np.array([offset_x, offset_y], dtype=np.float32)
        return [np.asarray(c, dtype=np.float32).reshape(1, 4, 2) * inv + shift for c in corners]

    def _detect_on_patch(
        self,
        gray_patch: np.ndarray,
        offset_x: float,
        offset_y: float,
        *,
        patch_scale: float = 1.0,
    ) -> tuple[list, np.ndarray, str]:
        scale = max(1.0, float(patch_scale))
        work = gray_patch
        if scale > 1.0:
            h, w = gray_patch.shape[:2]
            work = cv2.resize(
                gray_patch,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        prepared = _prepare_gray(work, self._tag)
        corners, ids, method = self._detect_apriltag(prepared)
        if not len(ids):
            return [], ids, method
        mapped = self._map_patch_corners(corners, offset_x, offset_y, scale)
        return mapped, ids, method

    def _roi_from_corners(self, corners: list, w: int, h: int) -> tuple[int, int, int, int] | None:
        if not corners:
            return None
        pts = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in corners], axis=0)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        bw, bh = x1 - x0, y1 - y0
        pad_frac = float(self._tag.roi_pad_frac)
        pad = max(float(self._tag.roi_min_size_px) * 0.5, max(bw, bh) * pad_frac)
        rx0 = max(0, int(x0 - pad))
        ry0 = max(0, int(y0 - pad))
        rx1 = min(w, int(x1 + pad))
        ry1 = min(h, int(y1 + pad))
        if rx1 - rx0 < 16 or ry1 - ry0 < 16:
            return None
        return rx0, ry0, rx1, ry1

    def _detect_full_frame(self, gray: np.ndarray) -> tuple[list, np.ndarray, str]:
        h, w = gray.shape[:2]
        max_w = int(self._tag.max_detection_width)
        scale_back = 1.0
        work = gray
        if max_w > 0 and w > max_w:
            scale_back = max_w / float(w)
            work = cv2.resize(
                gray,
                (max_w, max(1, int(h * scale_back))),
                interpolation=cv2.INTER_AREA,
            )
        prepared = _prepare_gray(work, self._tag)
        corners, ids, method = self._detect_apriltag(prepared)
        if scale_back != 1.0 and len(ids):
            inv = 1.0 / scale_back
            corners = [np.asarray(c, dtype=np.float32) * inv for c in corners]
        return corners, ids, method

    def detect(self, frame_bgr: np.ndarray) -> FiducialDetection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape[:2]

        corners: list = []
        ids = np.empty((0, 1), dtype=np.int32)
        method = "none"
        roi_upscale = max(1.0, float(self._tag.roi_upscale))

        # Fixed landing-pad ROI: full-res crop + optional upscale (best for fisheye).
        fixed_roi = self._norm_roi_pixels(gw, gh)
        if fixed_roi is not None:
            rx0, ry0, rx1, ry1 = fixed_roi
            sub = gray[ry0:ry1, rx0:rx1]
            corners, ids, method = self._detect_on_patch(
                sub,
                float(rx0),
                float(ry0),
                patch_scale=roi_upscale,
            )

        # Fast path: search ROI around last detection first.
        if not len(ids) and self._tag.roi_tracking and self._last_roi is not None:
            rx0, ry0, rx1, ry1 = self._last_roi
            rx0, ry0 = max(0, min(rx0, gw - 1)), max(0, min(ry0, gh - 1))
            rx1, ry1 = max(rx0 + 1, min(rx1, gw)), max(ry0 + 1, min(ry1, gh))
            sub = gray[ry0:ry1, rx0:rx1]
            corners, ids, method = self._detect_on_patch(
                sub,
                float(rx0),
                float(ry0),
                patch_scale=roi_upscale,
            )

        # Full frame if ROI found nothing (or disabled).
        if not len(ids):
            corners, ids, method = self._detect_full_frame(gray)

        corners, ids = self._dedupe_by_id(corners, ids)
        corners, ids = self._validate_layout(corners, ids)
        if len(ids) and method == "none":
            method = "apriltag"

        if self._tag.roi_tracking:
            if len(ids):
                self._last_roi = self._roi_from_corners(corners, gw, gh)
            else:
                self._last_roi = None

        return FiducialDetection(corners=corners, ids=ids, method=method)
