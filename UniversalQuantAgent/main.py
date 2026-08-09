"""Command-line entry point for Universal Quant Agent Version 1."""

from __future__ import annotations

from typing import Any, Callable

from modules.analyzer import rank_opportunities
from modules.finance import analyze_stock
from modules.sports import compare_teams
from modules.utils import print_key_values, print_section


def run_safely(label: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    """Run an external-data operation while keeping the rest of the app usable."""
    try:
        print(f"Fetching {label} data...")
        return operation()
    except Exception as error:  # API/network errors vary by provider.
        print(f"Could not complete {label} analysis: {error}")
        return None


def print_results(
    finance_insights: dict[str, Any] | None,
    sports_insights: dict[str, Any] | None,
    opportunities: list[dict[str, Any]],
) -> None:
    """Present results without mixing display code into analysis modules."""
    if finance_insights:
        print_section(f"Finance: {finance_insights['subject']}")
        print_key_values(finance_insights["summary"])
        print("  Patterns:")
        for pattern in finance_insights["patterns"]:
            print(f"    - {pattern['name'].replace('_', ' ').title()}: {pattern['description']}")
        print("  Correlations:")
        print_key_values(finance_insights["correlations"])

    if sports_insights:
        print_section(f"NBA: {sports_insights['subject']} ({sports_insights['season']})")
        for team in sports_insights["teams"]:
            print(f"  {team['name']}: {team['wins']}-{team['losses']}, "
                  f"Net Rating {team['net_rating']:+.1f}, TS% "
                  f"{team['efficiency']['true_shooting_pct']:.1f}")
        print(f"  V1 matchup lean: {sports_insights['matchup']['favored_team']}")

    print_section("Ranked Opportunities")
    if not opportunities:
        print("  No opportunities could be scored. Check the messages above.")
    for rank, item in enumerate(opportunities, start=1):
        print(f"  {rank}. [{item['score']:5.1f}/100] {item['opportunity']} ({item['domain']})")
        print(f"     {item['reason']}")


def main() -> None:
    """Collect user inputs and orchestrate all Version 1 modules."""
    print("Universal Quant Agent - Version 1")
    print("Educational analysis only; this is not financial or betting advice.\n")
    ticker = input("Stock ticker (example: AAPL): ").strip() or "AAPL"
    team_one = input("First NBA team (example: DEN): ").strip() or "DEN"
    team_two = input("Second NBA team (example: BOS): ").strip() or "BOS"
    print()

    finance_insights = run_safely("finance", lambda: analyze_stock(ticker))
    sports_insights = run_safely("sports", lambda: compare_teams(team_one, team_two))
    opportunities = rank_opportunities(finance_insights, sports_insights)
    print_results(finance_insights, sports_insights, opportunities)

    # FUTURE EXPANSION POINT: add sentiment, ML predictions, databases, APIs,
    # dashboards, and additional sports by calling their modules here and passing
    # their dictionary results into an expanded analyzer.


if __name__ == "__main__":
    main()

