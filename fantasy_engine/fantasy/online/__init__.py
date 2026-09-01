"""fantasy.online: the fantasy engine's only outbound-network touch points.

Everything here is opt-in and gracefully degrading: nothing imports these
modules at load time, no engine calls them, and each one returns an empty /
unchanged result on any failure. The engines themselves stay fully offline --
they read the local JSON files these modules write.
"""
