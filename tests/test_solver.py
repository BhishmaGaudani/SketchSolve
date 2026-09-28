import pytest

from src.solver import add_implicit_multiplication, resolve_x, solve_symbols, tokenize


def solve(text):
    """Helper: "2x+3=11" -> list of classifier labels, then solve."""
    return solve_symbols(list(text))


# ---- × vs x from context ----

@pytest.mark.parametrize("text, expected", [
    ("3x4", "3×4"),        # number on both sides: times
    ("2x+3", "2x+3"),      # followed by an operator: variable
    ("x=5", "x=5"),        # at the start: variable
    ("3x", "3x"),          # at the end: variable
    ("12x34", "12×34"),
    ("2x3=x", "2×3=x"),    # first is times, last is the variable
    ("(2)x(3)", "(2)×(3)"),
])
def test_resolve_x(text, expected):
    assert "".join(resolve_x(list(text))) == expected


# ---- building the expression string ----

def test_tokenize_groups_digits_and_maps_operators():
    assert tokenize(list("12×3÷4")) == ["12", "*", "3", "/", "4"]


def test_tokenize_drops_leading_zeros():
    assert tokenize(list("007")) == ["7"]


@pytest.mark.parametrize("tokens, expected", [
    (["2", "x"], "2*x"),
    (["x", "2"], "x*2"),
    (["x", "x"], "x*x"),
    (["2", "(", "x", ")"], "2*(x)"),
    ([")", "("], ")*("),
    (["2", "+", "x"], "2+x"),   # nothing added next to an operator
])
def test_implicit_multiplication(tokens, expected):
    assert "".join(add_implicit_multiplication(tokens)) == expected


# ---- arithmetic (no "=") ----

@pytest.mark.parametrize("text, answer", [
    ("3+5", "8"),
    ("7-10", "-3"),
    ("6x7", "42"),          # x between numbers is ×
    ("6×7", "42"),
    ("8÷2", "4"),
    ("7÷2", "7/2 ≈ 3.5"),   # exact fraction plus decimal
    ("2+3x4", "14"),        # multiplication before addition
    ("-3+5", "2"),          # leading minus
    ("3×-2", "-6"),         # minus right after an operator
    ("12+30", "42"),        # multi-digit numbers
])
def test_arithmetic(text, answer):
    result = solve(text)
    assert result.error is None
    assert result.answer == answer


# ---- equations (with "=") ----

@pytest.mark.parametrize("text, answer", [
    ("2x+3=11", "x = 4"),
    ("x=5", "x = 5"),
    ("5=x", "x = 5"),
    ("3x=7", "x = 7/3 ≈ 2.33333"),
    ("x-4=-6", "x = -2"),
    ("12÷x=4", "x = 3"),
    ("xx=9", "x = -3 or x = 3"),
    ("2x3=x", "x = 6"),
    ("x+1=1+x", "True for every x"),
    ("x+1=x+2", "No solution"),
    ("3+4=7", "True"),
    ("3+4=8", "False (left side is 7, right side is 8)"),
])
def test_equations(text, answer):
    result = solve(text)
    assert result.error is None
    assert result.answer == answer


def test_result_shows_expression_and_parsed_string():
    result = solve("2x+3=11")
    assert result.expression == "2x+3=11"
    assert result.parsed == "2*x+3=11"


def test_times_sign_shown_in_expression():
    assert solve("3x4").expression == "3×4"


# ---- errors: a message, never a crash ----

@pytest.mark.parametrize("text, message_part", [
    ("", "Nothing to solve"),
    ("3+", "ends with an operator"),
    ("+3", "unexpected place"),
    ("3++4", "unexpected place"),
    ("3××4", "unexpected place"),   # would be 3**4 (power) if not caught
    ("3--4", "unexpected place"),
    ("=5", "left side of '=' is empty"),
    ("5=", "right side of '=' is empty"),
    ("1=2=3", "Only one '='"),
    ("2x+3", "no '='"),
    ("5÷0", "Division by zero"),
    ("0÷0", "Division by zero"),
    ("x÷0=1", "Division by zero"),
    ("(3+4", "never closed"),
    ("3+4)", "no matching"),
    ("÷", "unexpected place"),
])
def test_errors(text, message_part):
    result = solve(text)
    assert result.answer is None
    assert message_part in result.error


def test_unknown_characters_are_rejected_before_parsing():
    # If a future class label slipped through unmapped, it must never reach
    # parse_expr (which uses eval internally).
    result = solve_symbols(["2", "+", "__import__('os')"])
    assert result.answer is None
    assert result.error is not None
