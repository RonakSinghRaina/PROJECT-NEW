import json
import tokenize
import io

NOTEBOOK_PATH = r"cartoon_analysis_complete_new.ipynb"

def remove_comments_from_code(source_code):
    """Remove comments from Python code using tokenize, preserving everything else."""
    if not source_code.strip():
        return source_code

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source_code).readline))
    except tokenize.TokenError:
        return source_code  # If tokenizing fails, return unchanged

    # Collect comment token positions
    comment_tokens = [t for t in tokens if t.type == tokenize.COMMENT]
    if not comment_tokens:
        return source_code

    lines = source_code.splitlines(True)  # keep line endings
    result_lines = list(lines)  # copy

    for ct in comment_tokens:
        row = ct.start[0] - 1  # 0-indexed
        col = ct.start[1]
        line = result_lines[row]

        before_comment = line[:col]

        # If the line is ONLY a comment (with optional leading whitespace), mark for removal
        if before_comment.strip() == '':
            result_lines[row] = None  # mark for deletion
        else:
            # Inline comment: strip trailing whitespace before comment, keep the code part
            result_lines[row] = before_comment.rstrip() + '\n'

    # Remove lines that were full-line comments, but preserve blank lines structure
    final_lines = [l for l in result_lines if l is not None]

    return ''.join(final_lines)


def process_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed_cells = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue

        source_lines = cell['source']
        original = ''.join(source_lines)

        new_code = remove_comments_from_code(original)

        if new_code != original:
            # Preserve the notebook's line-by-line format
            if new_code:
                split = new_code.splitlines(True)
                # Last line should not have trailing newline (notebook convention)
                if split and split[-1].endswith('\n'):
                    split[-1] = split[-1][:-1]
                cell['source'] = split
            else:
                cell['source'] = []
            changed_cells += 1

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Done. Modified {changed_cells} code cells.")


if __name__ == '__main__':
    process_notebook(NOTEBOOK_PATH)
