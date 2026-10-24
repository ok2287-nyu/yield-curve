"""Ego-centric bird's-eye-view renderer.

Output: uint8 image (3, H, W), channels as in PPUU:
    0 (red)   road: the aerial photo in grayscale, warped into the ego frame
    1 (green) every other road user, drawn as a rectangle at its true size and heading
    2 (blue)  the ego car (always in the same place, pointing up)

Ego frame: origin at the ego centre, "forward" along the ego heading, "left" 90 degrees CCW from forward.
Image layout: forward = up (towards row 0), left = towards column 0.
    row = ego_row - forward / m_per_px
    col = ego_col - left / m_per_px
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import RenderConfig


def car_corners(x, y, heading, length, width):
    """World-frame corners (N, 4, 2) of N rectangles."""
    x, y, heading, length, width = (np.atleast_1d(np.asarray(a, dtype=np.float64))
                                    for a in (x, y, heading, length, width))
    c, s = np.cos(heading), np.sin(heading)
    fwd = np.stack([c, s], -1)            # (N, 2)
    left = np.stack([-s, c], -1)
    hl, hw = (length / 2)[:, None], (width / 2)[:, None]
    centre = np.stack([x, y], -1)
    corners = [centre + hl * fwd + hw * left, centre + hl * fwd - hw * left,
               centre - hl * fwd - hw * left, centre - hl * fwd + hw * left]
    return np.stack(corners, axis=1)


def world_to_ego_px(points, ego_x, ego_y, ego_heading, rc: RenderConfig):
    """World points (..., 2) -> (col, row) pixel coordinates (..., 2) in the ego image."""
    d = points - np.array([ego_x, ego_y])
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    forward = d[..., 0] * c + d[..., 1] * s
    left = -d[..., 0] * s + d[..., 1] * c
    col = rc.ego_col - left / rc.m_per_px
    row = rc.ego_row - forward / rc.m_per_px
    return np.stack([col, row], -1)


def _fill_rects(px_corners, rc: RenderConfig) -> np.ndarray:
    """Rasterise rectangles (N, 4, 2) given in output-pixel coordinates; soft edges via supersampling."""
    k = rc.supersample
    canvas = np.zeros((rc.size * k, rc.size * k), np.uint8)
    if len(px_corners):
        # pixel centres sit at integer coordinates; shift by +0.5 so supersampled cells line up
        pts = np.round((px_corners + 0.5) * k - 0.5).astype(np.int32)
        cv2.fillPoly(canvas, list(pts), 255)
    return cv2.resize(canvas, (rc.size, rc.size), interpolation=cv2.INTER_AREA)


class RoadMap:
    """Grayscale road raster for one location, sampled into the ego frame with a single affine warp."""

    def __init__(self, background_path: str, m_per_px: float, rc: RenderConfig):
        img = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(background_path)
        # Pre-shrink once to the output resolution (area averaging = no aliasing),
        # so each per-frame warp is close to 1:1 and bilinear is enough.
        f = m_per_px / rc.m_per_px
        self.image = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        self.m_per_px = m_per_px / f   # metres per pixel of the pre-shrunk image

    def crop(self, ego_x, ego_y, ego_heading, rc: RenderConfig) -> np.ndarray:
        # Affine map: output pixel (col, row) -> background pixel (col_bg, row_bg).
        # world = ego + forward*(c, s) + left*(-s, c),  forward = (ego_row-row)*m,  left = (ego_col-col)*m
        # col_bg = world_x / M,  row_bg = -world_y / M   (rows count downward)
        c, s = np.cos(ego_heading), np.sin(ego_heading)
        r = rc.m_per_px / self.m_per_px
        a_col = np.array([s * r, -c * r])          # d(col_bg)/d(col), d(col_bg)/d(row)
        a_row = np.array([c * r, s * r])           # d(row_bg)/d(col), d(row_bg)/d(row)
        # value at the ego pixel must be the ego position
        ego_bg = np.array([ego_x / self.m_per_px, -ego_y / self.m_per_px])
        b_col = ego_bg[0] - a_col @ np.array([rc.ego_col, rc.ego_row])
        b_row = ego_bg[1] - a_row @ np.array([rc.ego_col, rc.ego_row])
        M = np.array([[a_col[0], a_col[1], b_col], [a_row[0], a_row[1], b_row]], np.float64)
        return cv2.warpAffine(self.image, M, (rc.size, rc.size),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def render_frame(ego, others, road: RoadMap | None, rc: RenderConfig) -> np.ndarray:
    """One ego-centric frame.

    ego:    dict with x, y, heading, length, width
    others: dict of arrays x, y, heading, length, width (other road users present in this frame)
    """
    img = np.zeros((3, rc.size, rc.size), np.uint8)
    if road is not None:
        img[0] = road.crop(ego["x"], ego["y"], ego["heading"], rc)

    if len(others["x"]):
        corners = car_corners(others["x"], others["y"], others["heading"], others["length"], others["width"])
        px = world_to_ego_px(corners, ego["x"], ego["y"], ego["heading"], rc)
        # skip vehicles entirely outside the crop (with a margin) to save rasterisation work
        keep = ((px[..., 0].max(1) > -2) & (px[..., 0].min(1) < rc.size + 2) &
                (px[..., 1].max(1) > -2) & (px[..., 1].min(1) < rc.size + 2))
        img[1] = _fill_rects(px[keep], rc)

    ego_corners = car_corners(ego["x"], ego["y"], ego["heading"], ego["length"], ego["width"])
    img[2] = _fill_rects(world_to_ego_px(ego_corners, ego["x"], ego["y"], ego["heading"], rc), rc)
    return img
