"""Regression tests for the Section-1 compound bug: phantom sliver boxes and the
merge-diluted staff verdict that caused confirmed staff-680 to be counted as a customer."""

from src.staff.filter import is_staff_track
from src.track.base import Track
from src.zones.roi import in_zone

# Whole-frame ROI so only the aspect gate decides in these tests.
_ROI = [(0, 0), (1280, 0), (1280, 720), (0, 720)]


def test_staff_dominant_segment_survives_merge_dilution():
    # A real staff track whose dominant 500-frame segment shows the uniform 72% of the
    # time, merged with a 300-frame segment where it doesn't (turned away / a pooled-in
    # phantom): pooled fraction is 360/800 = 0.45 < 0.7 and the old logic missed it, but
    # the dominant segment (0.72) is majority-staff, so it must still flag.
    assert is_staff_track([(360, 500), (0, 300)], 0.7) is True


def test_customer_not_flagged_by_a_short_staff_blip():
    # A customer whose dominant segment isn't staff stays a customer even if a tiny
    # 3-frame fragment spuriously matched the stripe.
    assert is_staff_track([(0, 500), (3, 3)], 0.7) is False


def test_in_zone_rejects_degenerate_sliver_box():
    person = Track(track_id=1, x1=700, y1=300, x2=760, y2=540, confidence=0.9)  # 60x240, w/h 0.25
    sliver = Track(track_id=2, x1=715, y1=320, x2=719, y2=554, confidence=0.9)  # 4x234, w/h 0.017
    assert in_zone(person, _ROI, box_depth_frac=0.4, min_box_aspect=0.1) is True
    assert in_zone(sliver, _ROI, box_depth_frac=0.4, min_box_aspect=0.1) is False
