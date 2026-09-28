"""Turn predicted symbols into an expression and solve it with SymPy.

    ["2", "x", "+", "3", "=", "1", "1"]  ->  "2*x + 3 = 11"  ->  x = 4

Steps:
  1. decide what each x-shaped symbol means: × or the variable x
  2. group digits into numbers and add the implicit "*" in "2x"
  3. check the structure (operators in valid places, one "=" at most) so the
     user gets a clear message instead of a parser error
  4. parse with SymPy and either evaluate (no "=") or solve for x (with "=")

Safety: sympy.parse_expr uses Python's eval() internally, so it must never
see arbitrary text. Here its input is built only from the model's fixed set
of labels, and ALLOWED_CHARS is checked right before parsing as a second guard.
"""
from dataclasses import dataclass

import sympy
from sympy.parsing.sympy_parser import parse_expr

X = sympy.Symbol("x")
OPERATORS = {"+", "-", "*", "/"}
ALLOWED_CHARS = set("0123456789x+-*/()")
# What each classifier label becomes in the expression string.
TO_PYTHON = {"×": "*", "÷": "/"}


class SolveError(Exception):
    """An expression we can't evaluate; the message is shown to the user."""


@dataclass
class SolveResult:
    expression: str          # what we think the user wrote, e.g. "2x+3=11"
    parsed: str = ""         # the string given to SymPy, e.g. "2*x+3=11"
    answer: str | None = None
    error: str | None = None


def resolve_x(symbols: list[str]) -> list[str]:
    """Decide whether each x-shaped symbol is × or the variable x.

    The model can't tell them apart by shape, so we use the neighbors: an x
    with a number (or bracket) directly on BOTH sides is a times sign
    ("3x4" -> 3×4). Anywhere else it's the variable ("2x+3", "x=5").
    """
    resolved = list(symbols)
    for i, sym in enumerate(symbols):
        if sym != "x":
            continue
        left = symbols[i - 1] if i > 0 else None
        right = symbols[i + 1] if i + 1 < len(symbols) else None
        if left is not None and right is not None and (left.isdigit() or left == ")") \
                and (right.isdigit() or right == "("):
            resolved[i] = "×"
    return resolved


def tokenize(symbols: list[str]) -> list[str]:
    """Group digits into numbers and translate symbols to Python operators.

    ["1", "2", "×", "x"] -> ["12", "*", "x"]
    """
    tokens = []
    for sym in symbols:
        sym = TO_PYTHON.get(sym, sym)
        if sym.isdigit() and tokens and tokens[-1].isdigit():
            tokens[-1] += sym           # continue the current number
        else:
            tokens.append(sym)
    # "007" is invalid Python syntax; normalize to "7".
    return [str(int(t)) if t.isdigit() else t for t in tokens]


def is_operand_end(token: str) -> bool:
    return token.isdigit() or token in ("x", ")")


def is_operand_start(token: str) -> bool:
    return token.isdigit() or token in ("x", "(")


def add_implicit_multiplication(tokens: list[str]) -> list[str]:
    """Insert "*" where math leaves it out: "2x" -> "2*x", "x(" -> "x*(" ..."""
    result = []
    for token in tokens:
        if result and is_operand_end(result[-1]) and is_operand_start(token):
            result.append("*")
        result.append(token)
    return result


def check_side(tokens: list[str], side_name: str) -> None:
    """Raise SolveError if one side of the equation isn't well-formed."""
    if not tokens:
        raise SolveError(f"The {side_name} of '=' is empty.")

    depth = 0
    prev = None  # previous token, None at the start
    for token in tokens:
        if token in OPERATORS:
            after_operand = prev is not None and is_operand_end(prev)
            # A "-" may also start a number: at the start, after "(", or
            # after another operator ("3*-2"). Any other operator needs
            # something before it. This also stops "3××4" becoming "3**4",
            # which Python would read as 3 to the power of 4.
            unary_minus = token == "-" and (prev is None or prev == "(" or prev in OPERATORS) \
                and prev != "-"
            if not (after_operand or unary_minus):
                shown = {"*": "×", "/": "÷"}.get(token, token)
                raise SolveError(f"'{shown}' is in an unexpected place.")
        elif token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                raise SolveError("A ')' has no matching '('.")
        prev = token

    if depth > 0:
        raise SolveError("A '(' is never closed.")
    if not is_operand_end(tokens[-1]):
        raise SolveError(f"The {side_name} ends with an operator.")


def to_sympy(tokens: list[str]) -> sympy.Expr:
    text = "".join(tokens)
    if not set(text) <= ALLOWED_CHARS:  # second guard before parse_expr (see top of file)
        raise SolveError("The expression contains unexpected characters.")
    try:
        return parse_expr(text, local_dict={"x": X})
    except (SyntaxError, TypeError, sympy.SympifyError):
        raise SolveError("Couldn't read this as a math expression.")


def format_number(value: sympy.Expr) -> str:
    """7 -> "7",  7/2 -> "7/2 ≈ 3.5",  2*sqrt(2) -> "2*sqrt(2) ≈ 2.82843".

    SymPy keeps exact fractions (7/2 stays 7/2) because all our numbers are
    integers, so there's no floating-point rounding until we format.
    """
    if value.is_Integer:
        return str(value)
    if value.is_real:
        return f"{value} ≈ {float(value):.6g}"
    return str(value)  # e.g. complex solutions like 2*I


def has_bad_value(value: sympy.Expr) -> bool:
    """True for SymPy's results of dividing by zero (zoo, nan, ...)."""
    return value.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo)


def evaluate(tokens: list[str]) -> str:
    """No "=": evaluate the arithmetic, e.g. 3+5 -> 8."""
    if "x" in tokens:
        raise SolveError("There's an x but no '=', so there's nothing to solve for.")
    check_side(tokens, "expression")
    value = to_sympy(tokens)
    if has_bad_value(value):
        raise SolveError("Division by zero.")
    return format_number(value)


def solve_equation(left: list[str], right: list[str]) -> str:
    """With "=": solve for x, or check the equation if there's no x."""
    check_side(left, "left side")
    check_side(right, "right side")
    lhs, rhs = to_sympy(left), to_sympy(right)
    if has_bad_value(lhs) or has_bad_value(rhs):
        raise SolveError("Division by zero.")

    # Check the sides as written: in "x+1=x+2" the x cancels out when
    # subtracting, but it's still an equation in x (with no solution).
    has_x = X in lhs.free_symbols or X in rhs.free_symbols
    difference = sympy.simplify(lhs - rhs)

    if not has_x:  # e.g. "3+4=7": just check whether it's true
        if difference == 0:
            return "True"
        return f"False (left side is {format_number(lhs)}, right side is {format_number(rhs)})"

    if difference == 0:  # e.g. "x+1=1+x"
        return "True for every x"
    solutions = sympy.solve(sympy.Eq(lhs, rhs), X)
    if not solutions:
        return "No solution"
    return " or ".join(f"x = {format_number(s)}" for s in solutions)


def solve_symbols(symbols: list[str]) -> SolveResult:
    """Main entry point: classifier labels in, answer (or error message) out."""
    resolved = resolve_x(symbols)
    result = SolveResult(expression="".join(resolved))
    try:
        if not symbols:
            raise SolveError("Nothing to solve.")
        tokens = add_implicit_multiplication(tokenize(resolved))
        result.parsed = "".join(tokens)

        if tokens.count("=") > 1:
            raise SolveError("Only one '=' is allowed.")
        if "=" in tokens:
            i = tokens.index("=")
            result.answer = solve_equation(tokens[:i], tokens[i + 1:])
        else:
            result.answer = evaluate(tokens)
    except SolveError as e:
        result.error = str(e)
    return result
