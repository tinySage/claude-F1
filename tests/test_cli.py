"""The ``pitwall`` command line parses every documented command."""

import pytest

from f1strat.cli import build_parser


@pytest.mark.parametrize(
    "argv",
    [
        ["data"],
        ["clean"],
        ["fit"],
        ["fit", "splits"],
        ["thin"],
        ["backtest"],
        ["backtest", "rehearsal", "spain"],
        ["case-study", "hungary"],
        ["case-study", "silverstone", "--hindsight-only"],
        ["case-study", "monaco"],
        ["figures", "--no-notebooks"],
    ],
)
def test_commands_parse(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_unknown_case_study_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["case-study", "monza"])
