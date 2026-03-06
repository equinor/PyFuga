"""
Test that all @jit decorator signatures match their function signatures.

This prevents runtime TypeError from Numba when the decorator specifies
a different number of arguments than the actual function has.
"""

import ast
from pathlib import Path

import pytest


def parse_numba_signature(sig_string):
    """
    Parse a Numba signature string and count the number of input arguments.

    Examples:
        "double(double, double)" -> 2
        "double(double,double,double,double)" -> 4
        "Tuple((double, double))(int32, double)" -> 2
        "complex128[:,:](double, double, double)" -> 3

    Returns:
        Number of input arguments, or None if signature cannot be parsed.
    """
    if not isinstance(sig_string, str):
        return None

    # Find the last opening parenthesis (input arguments start there)
    # For "return_type(arg1, arg2)", we want "(arg1, arg2)"
    # For "Tuple((ret1, ret2))(arg1, arg2)", we want "(arg1, arg2)"

    # Count closing parens from the end to find the input args section
    paren_depth = 0
    last_open_paren = -1

    for i in range(len(sig_string) - 1, -1, -1):
        if sig_string[i] == ")":
            paren_depth += 1
        elif sig_string[i] == "(":
            paren_depth -= 1
            if paren_depth == 0:
                last_open_paren = i
                break

    if last_open_paren == -1:
        return None

    # Extract the input arguments part
    args_str = sig_string[last_open_paren + 1 : -1]

    if not args_str.strip():
        # Empty signature means no arguments
        return 0

    # Count arguments by counting commas at depth 0
    # This handles nested types like "complex128[:,:]"
    depth = 0
    arg_count = 1  # Start at 1 since we have at least one arg

    for char in args_str:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            arg_count += 1

    return arg_count


def get_function_param_count(func_node):
    """
    Count the number of parameters in a function definition AST node.

    Excludes self since it's implicit in methods.
    """
    args = func_node.args
    param_count = (
        len(args.args)
        + len(args.posonlyargs)
        + len(args.kwonlyargs)
        + (1 if args.vararg else 0)
        + (1 if args.kwarg else 0)
    )

    # Check if first parameter is 'self' (for methods)
    if args.args and args.args[0].arg == "self":
        param_count -= 1

    return param_count


def find_jit_decorated_functions(file_path):
    """
    Find all functions decorated with @jit in a Python file.

    Returns:
        List of tuples: (function_name, jit_signature, param_count, line_number)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
    except Exception:
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function has @jit decorator
            for decorator in node.decorator_list:
                # Handle @jit("signature")
                if isinstance(decorator, ast.Call):
                    if (
                        isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "jit"
                        and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)
                    ):
                        jit_sig = decorator.args[0].value
                        param_count = get_function_param_count(node)
                        results.append((node.name, jit_sig, param_count, node.lineno, str(file_path)))
                # Handle @jit (no arguments)
                elif isinstance(decorator, ast.Name) and decorator.id == "jit":
                    # No signature provided, so no mismatch possible
                    pass

    return results


def get_all_jit_functions():
    """
    Find all @jit decorated functions in the pyfuga package.

    Returns:
        List of tuples: (function_name, jit_signature, param_count, line_number, file_path)
    """
    pyfuga_dir = Path(__file__).parent.parent / "pyfuga"
    all_functions = []

    for py_file in pyfuga_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        functions = find_jit_decorated_functions(py_file)
        all_functions.extend(functions)

    return all_functions


def test_jit_signatures_match_function_params():
    """
    Test that all @jit decorator signatures match their function parameters.

    This prevents TypeError like:
        "Signature mismatch: 6 argument types given, but function takes 5 arguments"
    """
    all_functions = get_all_jit_functions()

    assert len(all_functions) > 0, "No @jit decorated functions found!"

    mismatches = []

    for func_name, jit_sig, param_count, line_no, file_path in all_functions:
        expected_args = parse_numba_signature(jit_sig)

        if expected_args is None:
            # Could not parse signature, skip
            continue

        if expected_args != param_count:
            mismatches.append(
                {
                    "function": func_name,
                    "file": file_path,
                    "line": line_no,
                    "jit_signature": jit_sig,
                    "expected_args": expected_args,
                    "actual_params": param_count,
                }
            )

    if mismatches:
        error_msg = "\n\nJIT signature mismatches found:\n\n"
        for m in mismatches:
            error_msg += (
                f"  {m['file']}:{m['line']}\n"
                f"    Function: {m['function']}\n"
                f"    JIT signature: {m['jit_signature']}\n"
                f"    Expected args: {m['expected_args']}\n"
                f"    Actual params: {m['actual_params']}\n\n"
            )
        pytest.fail(error_msg)


def test_parse_numba_signature():
    """Test the signature parser with various formats."""
    assert parse_numba_signature("double(double, double)") == 2
    assert parse_numba_signature("double(double,double,double,double)") == 4
    assert parse_numba_signature("double(double, double, double, double, double)") == 5
    assert parse_numba_signature("double(double, double, double, double, double, double)") == 6
    assert parse_numba_signature("Tuple((double, double))(int32, double)") == 2
    assert parse_numba_signature("complex128[:,:](double, double, double)") == 3
    assert parse_numba_signature("double[:](complex128[:,:],int32)") == 2
    assert parse_numba_signature("void()") == 0

    # Multi-line signatures with backslashes (should be cleaned before parsing)
    sig = "complex128[:,:](int32,double,double, complex128[:,:],complex128[:,:], double,int32)"
    assert parse_numba_signature(sig) == 7


if __name__ == "__main__":
    # For quick debugging
    test_parse_numba_signature()
    test_jit_signatures_match_function_params()
    print("All tests passed!")
