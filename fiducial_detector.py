"""
Detekcja markerów ArUco — podejście jak w ROS / fiducial:
  1) OpenCV ArUco3 (detectInvertedMarker, wieloprzebiegowy)
  2) Dopasowanie szablonu NCC (białe wzory na ciemnym tle)
  3) Walidacja geometryczna znanych ID 0–4
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import ArucoConfig


@dataclass
class FiducialDetection:
    corners: list  # list of (1,4,2) float32
    ids: np.ndarray  # (N,1) int32
    method: str  # aruco3 | template | none


def _get_dictionary(name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Nieznany słownik ArUco: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_aruco_params(ar_cfg: ArucoConfig) -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 33
    p.adaptiveThreshWinSizeStep = 8
    p.minMarkerPerimeterRate = float(ar_cfg.min_marker_perimeter_rate)
    p.maxMarkerPerimeterRate = 4.0
    p.minCornerDistanceRate = 0.05
    p.minDistanceToBorder = 3
    p.errorCorrectionRate = 0.5
    p.detectInvertedMarker = True
    if hasattr(p, "useAruco3Detection"):
        p.useAruco3Detection = bool(ar_cfg.use_aruco3_detection)
    refine = getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", cv2.aruco.CORNER_REFINE_SUBPIX)
    p.cornerRefinementMethod = refine
    return p


def _gray_variants(gray: np.ndarray, ar_cfg: ArucoConfig) -> list[np.ndarray]:
    out: list[np.ndarray] = [gray]
    if ar_cfg.use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out.append(clahe.apply(gray))
    eq = cv2.equalizeHist(gray)
    out.append(eq)
    return out


class FiducialDetector:
    """Jeden backend na klatkę — ArUco3 + szablony + limit ID."""

    def __init__(self, ar_cfg: ArucoConfig) -> None:
        self._ar = ar_cfg
        self._dict_name = ar_cfg.dictionary_name
        self._dictionary = _get_dictionary(self._dict_name)
        self._params = _make_aruco_params(ar_cfg)
        self._aruco = cv2.aruco.ArucoDetector(self._dictionary, self._params)
        self._allowed = sorted(int(x) for x in ar_cfg.marker_ids)
        self._allowed_set = set(self._allowed)
        self._templates = self._build_ncc_templates()
        self._body_xy = self._marker_centers_xy()

    @staticmethod
    def _marker_centers_xy() -> dict[int, tuple[float, float]]:
        hf, hl = 0.07, 0.07
        return {
            0: (0.0, 0.0),
            1: (hf, hl),
            2: (hf, -hl),
            3: (-hf, -hl),
            4: (-hf, hl),
        }

    def _build_ncc_templates(self) -> list[tuple[int, np.ndarray]]:
        base = 80
        scales = tuple(float(s) for s in self._ar.template_match_scales)
        out: list[tuple[int, np.ndarray]] = []
        for mid in self._allowed:
            try:
                inner = cv2.aruco.generateImageMarker(self._dictionary, mid, base, borderBits=1)
            except cv2.error:
                continue
            white_on_black = (255 - inner).astype(np.uint8)
            for s in scales:
                side = max(20, int(round(base * s)))
                tpl = cv2.resize(white_on_black, (side, side), interpolation=cv2.INTER_AREA)
                out.append((mid, tpl))
        return out

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
        cap = max(1, int(self._ar.max_markers_per_frame))
        ranked = sorted(best.items(), key=lambda x: x[1][1], reverse=True)[:cap]
        out_c = [x[1][0] for x in ranked]
        out_ids = np.array([[x[0]] for x in ranked], dtype=np.int32)
        return out_c, out_ids

    def _validate_layout(self, corners: list, ids: np.ndarray) -> tuple[list, np.ndarray]:
        """Odrzuca trafienia sprzeczne z typowym rozmiarem markera na dronie."""
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
        if len(keep_ids) >= 2:
            d01 = np.linalg.norm(centers.get(0, centers[keep_ids[0]]) - centers.get(1, centers[keep_ids[1]]))
            for a, b in ((1, 2), (0, 3)):
                if a in centers and b in centers:
                    d = float(np.linalg.norm(centers[a] - centers[b]))
                    if d01 > 10 and d > d01 * 4.5:
                        pass
        if not keep_ids:
            return corners, ids
        return keep_c, np.array([[x] for x in keep_ids], dtype=np.int32)

    def _detect_aruco3(self, gray: np.ndarray) -> tuple[list, np.ndarray]:
        best_c: list = []
        best_ids = np.empty((0, 1), dtype=np.int32)
        best_n = 0
        h, w = gray.shape[:2]
        scales = [1.0]
        if min(h, w) < 280:
            scales.append(2.0)
        for scale in scales:
            g = gray if scale == 1.0 else cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            inv = 1.0 / scale
            for variant in _gray_variants(g, self._ar):
                corners, ids, _ = self._aruco.detectMarkers(variant)
                corners, ids = self._filter_allowed(corners, ids)
                if len(ids) > best_n:
                    best_n = len(ids)
                    if scale != 1.0 and corners:
                        scaled_c = []
                        for c in corners:
                            c2 = np.array(c, dtype=np.float32, copy=True)
                            c2 *= inv
                            scaled_c.append(c2)
                        best_c, best_ids = scaled_c, ids
                    else:
                        best_c, best_ids = list(corners), ids
                if best_n >= 2:
                    break
            if best_n >= 2:
                break
        return best_c, best_ids

    def _detect_template_ncc(self, gray: np.ndarray) -> tuple[list, np.ndarray]:
        gh, gw = gray.shape[:2]
        thr = float(self._ar.template_match_threshold)
        margin = float(self._ar.template_match_margin)
        per_id: dict[int, tuple[float, int, int, int, int]] = {}

        for mid, tpl in self._templates:
            th, tw = tpl.shape[:2]
            if th > gh or tw > gw:
                continue
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (px, py) = cv2.minMaxLoc(res)
            prev = per_id.get(mid)
            if prev is None or peak > prev[0]:
                per_id[mid] = (float(peak), px, py, tw, th)

        if not per_id:
            return [], np.empty((0, 1), dtype=np.int32)

        ranked = sorted(per_id.items(), key=lambda x: x[1][0], reverse=True)
        best_val = ranked[0][1][0]
        corners: list = []
        ids_list: list[int] = []
        cap = int(self._ar.max_markers_per_frame)
        for mid, (peak, px, py, tw, th) in ranked:
            if peak < thr or best_val - peak > margin:
                continue
            c = np.array(
                [[[px, py], [px + tw, py], [px + tw, py + th], [px, py + th]]],
                dtype=np.float32,
            )
            corners.append(c)
            ids_list.append(mid)
            if len(ids_list) >= cap:
                break
        if not ids_list:
            return [], np.empty((0, 1), dtype=np.int32)
        return corners, np.array([[x] for x in ids_list], dtype=np.int32)

    def detect(self, frame_bgr: np.ndarray) -> FiducialDetection:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        max_w = int(self._ar.max_detection_width)
        scale_back = 1.0
        if max_w > 0 and w > max_w:
            scale_back = max_w / float(w)
            gray = cv2.resize(gray, (max_w, max(1, int(h * scale_back))), interpolation=cv2.INTER_AREA)

        corners, ids = self._detect_aruco3(gray)
        method = "aruco3" if len(ids) else "none"

        if len(ids) == 0 and self._ar.use_template_fallback:
            corners, ids = self._detect_template_ncc(gray)
            method = "template" if len(ids) else "none"

        if scale_back != 1.0 and len(ids):
            inv = 1.0 / scale_back
            corners = [np.array(c, dtype=np.float32) * inv for c in corners]

        corners, ids = self._dedupe_by_id(corners, ids)
        corners, ids = self._validate_layout(corners, ids)
        return FiducialDetection(corners=corners, ids=ids, method=method)
