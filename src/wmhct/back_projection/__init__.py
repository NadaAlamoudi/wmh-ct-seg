"""Stage 5: back projection into native FLAIR space.

Predictions are produced in native CT space. Before they are compared with the reference
they are carried into native FLAIR space by inverting the FLAIR-to-CT transform.
"""
