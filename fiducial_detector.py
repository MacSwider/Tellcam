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
            nthreads=2,
            quad_decimate=float(tag_cfg.quad_decimate),
            quad_sigma=float(tag_cfg.quad_sigma),
            refine_edges=1 if tag_cfg.refine_edges else 0,
            decode_sharpening=float(tag_cfg.decode_sharpening),
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if tag_cfg.use_clahe else None
        # Stan śledzenia ROI (wycinek wokół ostatniej detekcji).
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

    def _single_pass(self, gray: np.ndarray, margin_thr: float) -> tuple[list, list[int]]:
        """Jeden przebieg: raw, a jeśli pusto i CLAHE włączone — wariant CLAHE."""
        corners, ids_list = self._collect_detections(gray, margin_thr)
        if not ids_list and self._clahe is not None:
            corners, ids_list = self._collect_detections(self._clahe.apply(gray), margin_thr)
        return corners, ids_list

    def _detect_apriltag(self, gray: np.ndarray) -> tuple[list, np.ndarray, str]:
        """Lekka ścieżka: raw → CLAHE → opcjonalnie jeden upscale (tylko gdy skonfigurowany)."""
        h, w = gray.shape[:2]
        margin_thr = float(self._tag.min_decision_margin)

        corners, ids_list = self._single_pass(gray, margin_thr)
        if ids_list:
            return corners, np.array([[x] for x in ids_list], dtype=np.int32), "apriltag"

        # Opcjonalny pojedynczy upscale dla małych/dalekich tagów (domyślnie wyłączony).
        for scale in (s for s in self._tag.detection_upscales if float(s) > 1.0):
            inv = 1.0 / float(scale)
            g = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            corners, ids_list = self._single_pass(g, margin_thr)
            if ids_list:
                corners = [np.asarray(c, dtype=np.float32) * inv for c in corners]
                return corners, np.array([[x] for x in ids_list], dtype=np.int32), "apriltag"
            break  # tylko jedna próba upscalingu w gorącej pętli

        return [], np.empty((0, 1), dtype=np.int32), "none"

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

    def detect(self, frame_bgr: np.ndarray) -> FiducialDetection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        max_w = int(self._tag.max_detection_width)
        scale_back = 1.0
        if max_w > 0 and w > max_w:
            scale_back = max_w / float(w)
            gray = cv2.resize(gray, (max_w, max(1, int(h * scale_back))), interpolation=cv2.INTER_AREA)
        gh, gw = gray.shape[:2]

        corners: list = []
        ids = np.empty((0, 1), dtype=np.int32)
        method = "none"

        # Szybka ścieżka: szukaj najpierw w ROI wokół ostatniej detekcji.
        if self._tag.roi_tracking and self._last_roi is not None:
            rx0, ry0, rx1, ry1 = self._last_roi
            rx0, ry0 = max(0, min(rx0, gw - 1)), max(0, min(ry0, gh - 1))
            rx1, ry1 = max(rx0 + 1, min(rx1, gw)), max(ry0 + 1, min(ry1, gh))
            sub = gray[ry0:ry1, rx0:rx1]
            c_roi, ids_roi, m_roi = self._detect_apriltag(sub)
            if len(ids_roi):
                corners = [np.asarray(c, dtype=np.float32).reshape(1, 4, 2) + np.array([rx0, ry0], dtype=np.float32) for c in c_roi]
                ids, method = ids_roi, m_roi

        # Pełna klatka, jeśli ROI nic nie dało (lub wyłączone).
        if not len(ids):
            corners, ids, method = self._detect_apriltag(gray)

        if scale_back != 1.0 and len(ids):
            inv = 1.0 / scale_back
            corners = [np.asarray(c, dtype=np.float32) * inv for c in corners]

        corners, ids = self._dedupe_by_id(corners, ids)
        corners, ids = self._validate_layout(corners, ids)
        if len(ids) and method == "none":
            method = "apriltag"

        # Aktualizuj ROI na podstawie detekcji w przestrzeni gray (przed scale_back).
        if self._tag.roi_tracking:
            if len(ids):
                src = corners
                if scale_back != 1.0:
                    src = [np.asarray(c, dtype=np.float32) * scale_back for c in corners]
                self._last_roi = self._roi_from_corners(src, gw, gh)
            else:
                self._last_roi = None

        return FiducialDetection(corners=corners, ids=ids, method=method)
