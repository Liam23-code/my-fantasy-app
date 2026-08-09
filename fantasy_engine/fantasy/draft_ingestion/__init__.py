"""Draft-data ingestion modules, one per named source.

Read each module's own docstring before trusting its output -- they are not
uniformly "real, live, automatic" despite being named identically in the
architecture. Verified this session, against the actual endpoints, using both
a browsing fetch and a live call from this project's own Python runtime:

* :mod:`fantasy.draft_ingestion.ffcalculator` -- genuinely real, public, live.
  The only module in this package that performs an automatic network fetch.
* :mod:`fantasy.draft_ingestion.sleeper_api` -- checked live; Sleeper's public
  API has no ADP or draft-position data at all (only player identity dumps and
  waiver trending-add counts). Returns ``[]`` unconditionally, not a stub
  standing in for "not yet implemented".
* :mod:`fantasy.draft_ingestion.ffpc` -- checked live; FFPC's draft data is
  behind account login, no public page. Provides a real HTML-table parser for
  page content the user supplies themselves (e.g. saved after their own
  login), not automatic fetching.
* :mod:`fantasy.draft_ingestion.fp_analyzer` -- FantasyPros' Draft Analyzer
  CSV export requires their login and a completed draft session; this module
  is a real parser for that CSV format, fed a user-supplied file.
* :mod:`fantasy.draft_ingestion.underdog`, :mod:`fantasy.draft_ingestion.github_datasets`,
  :mod:`fantasy.draft_ingestion.reddit_dumps` -- no confirmed public,
  unauthenticated endpoint for any of these (the latter two also aren't a
  single identifiable source at all -- "a" GitHub dataset or Reddit post isn't
  a URL). Each provides a real, defensive parser for user-supplied content in
  a documented shape, returning ``[]`` when nothing is supplied.
"""
