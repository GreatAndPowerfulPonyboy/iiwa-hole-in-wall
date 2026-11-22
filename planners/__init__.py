"""Every planner of the benchmark, one per file.

hitw imports each ``planners/<name>.py`` as ``planners.<name>`` the first
time a planner is looked up, so a planner registered with
``@hitw.api.planner`` in any of these files is available by its id.
"""
