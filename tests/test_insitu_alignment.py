"""Every recognised measure must be assigned exactly once."""
import contextlib
import io
import unittest
from insitu.align import align


class AlignmentTest(unittest.TestCase):
    def test_split_and_drifting_systems_conserve_measures(self):
        for pdf, omr in [([2, 2], [1, 1, 2]), ([2, 2], [2, 4]),
                         ([3, 3, 3], [9]), ([2, 3], [1, 1])]:
            with self.subTest(pdf=pdf, omr=omr), contextlib.redirect_stdout(io.StringIO()):
                slices = align(pdf, omr)
                self.assertEqual(len(slices), len(pdf))
                indices = [i for s in slices for i in range(s.omr_start, s.omr_end)]
                self.assertEqual(indices, list(range(sum(omr))))
