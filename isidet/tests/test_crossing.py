"""Plain-python tests (repo has no pytest). Run: python isidet/tests/test_crossing.py"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.shared.crossing import CrossingDetector


def test_counts_clean_crossing_left_to_right():
    d = CrossingDetector()
    # line at x=100, 'after' side is x>100 (left_to_right)
    assert d.update([1], [90.0], 100.0, after_is_greater=True) == set()   # before
    assert d.update([1], [110.0], 100.0, after_is_greater=True) == {1}    # crossed
    print("ok clean crossing")


def test_counts_once_only():
    d = CrossingDetector()
    d.update([1], [90.0], 100.0, True)
    assert d.update([1], [110.0], 100.0, True) == {1}
    assert d.update([1], [120.0], 100.0, True) == set()   # already fired
    print("ok counts once")


def test_recovers_when_flip_frame_is_dropped():
    # No frame ever lands exactly at the line; track jumps 80 -> (gap) -> 130.
    d = CrossingDetector()
    assert d.update([7], [80.0], 100.0, True) == set()
    assert d.update([7], [130.0], 100.0, True) == {7}     # still counted
    print("ok recovers dropped flip")


def test_respects_belt_direction_right_to_left():
    # 'after' side is x<100 (right_to_left); moving from 130 -> 70 should count.
    d = CrossingDetector()
    assert d.update([3], [130.0], 100.0, after_is_greater=False) == set()
    assert d.update([3], [70.0], 100.0, after_is_greater=False) == {3}
    print("ok right_to_left")


def test_no_count_for_wrong_direction():
    # left_to_right line, object only ever on the after side, then wanders further
    # after — never seen before the line -> never counts (avoids phantom counts of
    # objects that enter already past the line).
    d = CrossingDetector()
    assert d.update([5], [110.0], 100.0, True) == set()
    assert d.update([5], [150.0], 100.0, True) == set()
    print("ok no wrong-direction count")


def test_two_close_tracks_both_count():
    d = CrossingDetector()
    d.update([1, 2], [90.0, 85.0], 100.0, True)
    assert d.update([1, 2], [110.0, 105.0], 100.0, True) == {1, 2}
    print("ok two close tracks")


def test_forget_prunes_state():
    d = CrossingDetector()
    d.update([1], [90.0], 100.0, True)
    d.update([1], [110.0], 100.0, True)
    d.forget(keep_ids={2})           # 1 no longer active
    # id 1 reused later as a brand-new track -> may count again (new physical parcel)
    d.update([1], [90.0], 100.0, True)
    assert d.update([1], [110.0], 100.0, True) == {1}
    print("ok forget prunes")


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print("ALL CROSSING TESTS PASSED")
