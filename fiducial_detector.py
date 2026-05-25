"""
Detekcja markerów AprilTag (rodzina tag36h11) — pupil_apriltags:
  wieloprzebiegowy (CLAHE / histogram), skalowanie dla małych klatek,
  fallback dla naklejek bez białej ramki (wycięcie czarnego kwadratu + pad),
  filtr ID 0–4, deduplikacja i walidacja geometryczna układu.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pupil_apriltags import Detector

from config import AprilTagConfig


@dataclass
class FiducialDetection:
    corners: list  # list of (1,4,2) float32 — kolejność zgodna z OpenCV solvePnP
    ids: np.ndarray  # (N,1) int32
    method: str  # apriltag | apriltag_crop | none


def _corners_to_opencv_order(corners: np.ndarray) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return c.reshape(1, 4, 2)


def _gray_variants(gray: np.ndarray, tag_cfg: AprilTagConfig) -> list[np.ndarray]:
    out: list[np.ndarray] = [gray]
    if tag_cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out.append(clahe.apply(gray))
    out.append(cv2.equalizeHist(gray))
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
    Wyodrębnia ciemny kwadrat tagu (bez czerwonej podkładki) i dodaje białą ramkę.
    Zwraca: patch, x0, y0, bw, bh, pad_px, upscale_factor.
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
    """Backend detekcji AprilTag na klatkę."""

    def __init__(self, tag_cfg: AprilTagConfig) -> None:
        self._tag = tag_cfg
        self._allowed = sorted(int(x) for x in tag_cfg.marker_ids)
        self._allowed_set = set(self._allowed)
        self._detector = Detector(
            families=tag_cfg.family,
            nthreads=1,
            quad_decimate=float(tag_cfg.quad_decimate),
            quad_sigma=float(tag_cfg.quad_sigma),
            refine_edges=1 if tag_cfg.refine_edges else 0,
            decode_sharpening=float(tag_cfg.decode_sharpening),
        )

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
            if med_area > 0 and areas[mid] < med_area * 0.15:
                continue
            if med_area > 0 and areas[mid] > med_area * 8.0:
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
        """Zbierz trafienia z bieżącej skali obrazu (opcjonalnie z patch → pełna klatka)."""
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

    def _detect_apriltag(self, gray: np.ndarray) -> tuple[list, np.ndarray, str]:
        best_c: list = []
        best_ids = np.empty((0, 1), dtype=np.int32)
        best_n = 0
        method = "none"
        h, w = gray.shape[:2]
        scales = [1.0]
        if min(h, w) < 280:
            scales.append(2.0)
        margin_thr = float(self._tag.min_decision_margin)

        for scale in scales:
            g = gray if scale == 1.0 else cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            inv = 1.0 / scale
            for variant in _gray_variants(g, self._tag):
                corners, ids_list = self._collect_detections(variant, margin_thr)
                if not ids_list:
                    continue
                ids_arr = np.array([[x] for x in ids_list], dtype=np.int32)
                if len(ids_list) > best_n:
                    best_n = len(ids_list)
                    if scale != 1.0:
                        scaled_c = [np.array(c, dtype=np.float32) * inv for c in corners]
                        best_c, best_ids = scaled_c, ids_arr
                    else:
                        best_c, best_ids = corners, ids_arr
                    method = "apriltag"
                if best_n >= 2:
                    return best_c, best_ids, method
            if best_n >= 2:
                return best_c, best_ids, method

        if best_n > 0:
            return best_c, best_ids, method

        if not self._tag.use_black_crop_fallback:
            return best_c, best_ids, method

        pad_frac = float(self._tag.black_crop_pad_frac)
        upscale = max(1, int(self._tag.black_crop_upscale))
        thresholds = tuple(int(t) for t in self._tag.black_crop_thresholds)

        for scale in scales:
            g = gray if scale == 1.0 else cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            inv = 1.0 / scale
            for variant in _gray_variants(g, self._tag):
                for thresh in thresholds:
                    info = _extract_black_tag_patch(variant, thresh, pad_frac)
                    if info is None:
                        continue
                    patch, x0, y0, _bw, _bh, pad_px, _ = info
                    up = cv2.resize(
                        patch,
                        (patch.shape[1] * upscale, patch.shape[0] * upscale),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    remap = (x0, y0, pad_px, upscale)
                    corners, ids_list = self._collect_detections(up, margin_thr, remap=remap)
                    if not ids_list:
                        continue
                    ids_arr = np.array([[x] for x in ids_list], dtype=np.int32)
                    if scale != 1.0:
                        corners = [np.array(c, dtype=np.float32) * inv for c in corners]
                    if len(ids_list) > best_n:
                        best_n = len(ids_list)
                        best_c, best_ids = corners, ids_arr
                        method = "apriltag_crop"
                    if best_n >= 1:
                        return best_c, best_ids, method

        return best_c, best_ids, method

    def detect(self, frame_bgr: np.ndarray) -> FiducialDetection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        max_w = int(self._tag.max_detection_width)
        scale_back = 1.0
        if max_w > 0 and w > max_w:
            scale_back = max_w / float(w)
            gray = cv2.resize(gray, (max_w, max(1, int(h * scale_back))), interpolation=cv2.INTER_AREA)

        corners, ids, method = self._detect_apriltag(gray)

        if scale_back != 1.0 and len(ids):
            inv = 1.0 / scale_back
            corners = [np.array(c, dtype=np.float32) * inv for c in corners]

        corners, ids = self._dedupe_by_id(corners, ids)
        corners, ids = self._validate_layout(corners, ids)
        if len(ids) and method == "none":
            method = "apriltag"
        return FiducialDetection(corners=corners, ids=ids, method=method)
