"""
logic_engine.py — Core violation detection engine

Analyses FrameDetections to:
1. Associate riders with each motorcycle (bounding box IoU / containment)
2. Associate helmets with each rider
3. Flag:
   - Triple riding  (>= TRIPLE_RIDING_THRESHOLD riders on one motorcycle)
   - Helmet violation (rider with no helmet)
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from config import OVERLAP_IOU_THRESHOLD, TRIPLE_RIDING_THRESHOLD, USE_CUSTOM_MODEL
from detector import Detection, FrameDetections

# --- Data Classes ---


@dataclass
class RiderGroup:
    """One motorcycle with its associated riders and helmet status."""

    motorcycle: Detection
    riders: List[Detection] = field(default_factory=list)
    helmeted_riders: List[Detection] = field(default_factory=list)
    helmetless_riders: List[Detection] = field(default_factory=list)

    @property
    def rider_count(self) -> int:
        return len(self.riders)

    @property
    def is_triple_riding(self) -> bool:
        return self.rider_count >= TRIPLE_RIDING_THRESHOLD

    @property
    def has_helmet_violation(self) -> bool:
        return len(self.helmetless_riders) > 0


@dataclass
class ViolationResult:
    """Outcome of analysing a single frame."""

    groups: List[RiderGroup] = field(default_factory=list)
    triple_riding_groups: List[RiderGroup] = field(default_factory=list)
    helmet_violation_groups: List[RiderGroup] = field(default_factory=list)
    lone_helmetless_riders: List[Detection] = field(
        default_factory=list
    )  # riders with no motorcycle match
    safe_groups: List[RiderGroup] = field(default_factory=list)

    @property
    def has_violation(self) -> bool:
        return bool(
            self.triple_riding_groups
            or self.helmet_violation_groups
            or self.lone_helmetless_riders
        )

    def violation_summary(self) -> str:
        parts = []
        if self.triple_riding_groups:
            parts.append(f"Triple Riding ×{len(self.triple_riding_groups)}")
        if self.helmet_violation_groups:
            parts.append(f"No Helmet ×{len(self.helmet_violation_groups)}")
        if self.lone_helmetless_riders:
            parts.append(f"Helmetless Rider ×{len(self.lone_helmetless_riders)}")
        return " | ".join(parts) if parts else "No Violation"

    def safe_summary(self) -> str:
        parts = []
        safe_single = sum(1 for g in self.safe_groups if g.rider_count == 1)
        safe_double = sum(1 for g in self.safe_groups if g.rider_count == 2)
        if safe_single:
            parts.append(f"Safe Single Ride ×{safe_single}")
        if safe_double:
            parts.append(f"Safe Double Ride ×{safe_double}")
        return " | ".join(parts) if parts else "Status: OK"


# --- IoU Helpers ---


def _iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute intersection-over-union between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(1, (xb2 - xb1) * (yb2 - yb1))
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _overlap_ratio(small_box: List[int], large_box: List[int]) -> float:
    """
    What fraction of small_box is covered by large_box?
    Used to check if a rider/helmet overlaps significantly with a motorcycle.
    """
    xs1, ys1, xs2, ys2 = small_box
    xl1, yl1, xl2, yl2 = large_box

    inter_x1 = max(xs1, xl1)
    inter_y1 = max(ys1, yl1)
    inter_x2 = min(xs2, xl2)
    inter_y2 = min(ys2, yl2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    small_area = max(1, (xs2 - xs1) * (ys2 - ys1))
    return inter_area / small_area


def _box_center(box: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# --- Association Logic ---


def _associate_riders_to_motorcycles(
    motorcycles: List[Detection], riders: List[Detection]
) -> Tuple[List[RiderGroup], List[Detection]]:
    """
    Assign each rider to the motorcycle with the highest overlap,
    provided overlap ratio >= OVERLAP_IOU_THRESHOLD.

    Returns:
        groups: list of RiderGroup (one per motorcycle)
        unmatched_riders: riders not assigned to any motorcycle
    """
    groups = [RiderGroup(motorcycle=m) for m in motorcycles]
    unmatched = []

    for rider in riders:
        best_score = OVERLAP_IOU_THRESHOLD
        best_group = None

        for g in groups:
            score = _overlap_ratio(rider.box, g.motorcycle.box)
            # Also consider expanded motorcycle zone (bikes + riders above them)
            mx1, my1, mx2, my2 = g.motorcycle.box
            expanded_box = [mx1, max(0, my1 - int((my2 - my1) * 1.2)), mx2, my2]
            score_exp = _overlap_ratio(rider.box, expanded_box)
            final_score = max(score, score_exp)

            if final_score > best_score:
                best_score = final_score
                best_group = g

        if best_group:
            best_group.riders.append(rider)
        else:
            unmatched.append(rider)

    return groups, unmatched


def _associate_helmets_to_riders(
    riders: List[Detection], helmets: List[Detection], no_helmets: List[Detection]
) -> Tuple[List[Detection], List[Detection]]:
    """
    Determine which riders have helmets and which don't.

    Strategy:
    - If custom model: class_id directly tells us (CLASS_HELMET vs CLASS_NO_HELMET)
    - If COCO fallback: helmet detection is unavailable; assume violation so demo still works

    Returns:
        helmeted, helmetless
    """
    if not USE_CUSTOM_MODEL:
        # Fallback: no helmet class available — mark all riders as helmetless for demo
        return [], list(riders)

    helmeted = []
    helmetless = []

    # Riders matched with a helmet bounding box (upper body overlap)
    matched_helmet_ids = set()

    for rider in riders:
        rx1, ry1, rx2, ry2 = rider.box
        head_box = [
            rx1,
            ry1,
            rx2,
            ry1 + int((ry2 - ry1) * 0.35),
        ]  # upper 35% = head region
        rider_has_helmet = False

        for h in helmets:
            if id(h) in matched_helmet_ids:
                continue
            overlap = _overlap_ratio(h.box, head_box)
            if overlap > 0.3 or _iou(h.box, head_box) > 0.2:
                rider_has_helmet = True
                matched_helmet_ids.add(id(h))
                break

        # Also honour explicit CLASS_NO_HELMET detections from model
        for nh in no_helmets:
            overlap = _overlap_ratio(nh.box, head_box)
            if overlap > 0.3:
                rider_has_helmet = False
                break

        if rider_has_helmet:
            helmeted.append(rider)
        else:
            helmetless.append(rider)

    return helmeted, helmetless


# --- Public API ---


def analyse_frame(fd: FrameDetections) -> ViolationResult:
    """
    Main entry point. Takes a FrameDetections object and returns a ViolationResult.

    Args:
        fd: FrameDetections from detector.py

    Returns:
        ViolationResult with all grouped violations
    """
    result = ViolationResult()

    # 1. Associate riders with motorcycles
    groups, unmatched_riders = _associate_riders_to_motorcycles(
        fd.motorcycles, fd.riders
    )

    # 2. For each group, associate helmets to riders
    for group in groups:
        helmeted, helmetless = _associate_helmets_to_riders(
            group.riders, fd.helmets, fd.no_helmets
        )
        group.helmeted_riders = helmeted
        group.helmetless_riders = helmetless

    # 3. Handle unmatched riders (no motorcycle detected nearby)
    _, lone_helmetless = _associate_helmets_to_riders(
        unmatched_riders, fd.helmets, fd.no_helmets
    )

    # 4. Build result
    result.groups = groups
    result.triple_riding_groups = [g for g in groups if g.is_triple_riding]
    result.helmet_violation_groups = [g for g in groups if g.has_helmet_violation]
    result.lone_helmetless_riders = lone_helmetless
    result.safe_groups = [
        g
        for g in groups
        if not g.is_triple_riding and not g.has_helmet_violation and g.rider_count > 0
    ]

    return result
