"""
note_assignment.py
Assign detected noteheads to treble/bass staves and filter false positives.
"""
import numpy as np
from config import CFG


def assign_notes_to_staves(noteheads, grand_staff_pairs, dy):
    """
    Assign each detected notehead to either the treble or bass staff
    of a grand staff pair based on y-coordinate relative to the midpoint.

    Returns two lists: treble_notes, bass_notes.
    Each note gets 'clef', 'pair_idx', 'system' fields added.
    """
    treble_notes = []
    bass_notes = []

    for note in noteheads:
        y_center = note['y_center']
        best_pair_idx = -1
        best_clef = 'treble'
        best_dist = float('inf')

        for pair_idx, (treble_sys, bass_sys) in enumerate(grand_staff_pairs):
            treble_center = (treble_sys[0] + treble_sys[4]) / 2.0
            bass_center = (bass_sys[0] + bass_sys[4]) / 2.0
            midpoint = (treble_sys[4] + bass_sys[0]) / 2.0

            top_limit = treble_sys[0] - dy * 5
            bot_limit = bass_sys[4] + dy * 5

            if top_limit <= y_center <= bot_limit:
                if y_center <= midpoint:
                    dist = abs(y_center - treble_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair_idx = pair_idx
                        best_clef = 'treble'
                else:
                    dist = abs(y_center - bass_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair_idx = pair_idx
                        best_clef = 'bass'

        if best_pair_idx >= 0:
            note['clef'] = best_clef
            note['pair_idx'] = best_pair_idx
            if best_clef == 'treble':
                note['system'] = grand_staff_pairs[best_pair_idx][0]
                treble_notes.append(note)
            else:
                note['system'] = grand_staff_pairs[best_pair_idx][1]
                bass_notes.append(note)

    return treble_notes, bass_notes


def filter_false_positive_notes(notes, dy, clef='treble'):
    """
    Remove likely false positive notes based on score and position relative
    to the staff.
    """
    filtered = []
    for note in notes:
        system = note.get('system')
        if system is None:
            filtered.append(note)
            continue

        y = note['y_center']
        score = note.get('score', 1.0)
        y_top = system[0]
        y_bot = system[4]

        if clef == 'treble':
            below_staff = y - y_bot
            above_staff = y_top - y
            ac = CFG.assignment
            if below_staff > dy * ac.treble_below_1[0] and score < ac.treble_below_1[1]:
                continue
            if below_staff > dy * ac.treble_below_2[0] and score < ac.treble_below_2[1]:
                continue
            if score < ac.treble_min_score:
                continue
            if above_staff > dy * ac.treble_above_2[0] and score < ac.treble_above_2[1]:
                continue
            if above_staff > dy * ac.treble_above_1[0] and score < ac.treble_above_1[1]:
                continue

        if clef == 'bass':
            above_staff = y_top - y
            below_staff = y - y_bot
            ac = CFG.assignment
            if above_staff > dy * ac.bass_above_1[0] and score < ac.bass_above_1[1]:
                continue
            if below_staff > dy * ac.bass_below_1[0] and score < ac.bass_below_1[1]:
                continue
            if score < ac.bass_min_score:
                continue
            if below_staff > dy * 0.5 and score < 0.90 and abs(score - 0.95) > 0.02 and abs(score - 1.0) > 0.02:
                continue
            if below_staff > dy * 1.5 and score < 0.70:
                continue
            if below_staff > dy * 0.3 and score < 0.65:
                continue

        filtered.append(note)

    return filtered
