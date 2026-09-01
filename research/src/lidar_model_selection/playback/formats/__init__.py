"""Explicit, non-interchangeable recording-format adapters.

Import adapters from their concrete modules.  Keeping this package initializer
dependency-free prevents legacy raw playback from importing MCAP/PyYAML code.
"""
