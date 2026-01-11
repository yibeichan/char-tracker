"""
Shared constants for the char-tracker pipeline.

This module contains constants that are used across multiple scripts
to avoid duplication and ensure consistency.
"""

# Main characters in Friends TV show (set for O(1) lookups)
MAIN_CHARACTERS = {'rachel', 'monica', 'chandler', 'joey', 'phoebe', 'ross'}

# Labels to skip during processing (non-human or unclear)
SKIP_LABELS = [
    'dk', 'not_human', 'background', 'unclear', 'junk',
    'not face', 'not clear', 'guest', 'guy on the wheelchair',
    'kid in the hospital', 'random kid'
]

# DK label prefix for unknown/uncertain faces
DK_LABEL_PREFIX = 'dk'
