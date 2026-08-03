from autoevolve.mutate.parsing import (
    apply_search_replace,
    parse_file_blocks,
    parse_search_replace,
)


def test_search_replace_round_trip():
    response = """<<<<<<< SEARCH src/main.py
value = 1
=======
value = 2
>>>>>>> REPLACE
"""
    blocks = parse_search_replace(response)
    assert blocks == [("src/main.py", "value = 1\n", "value = 2\n")]

    files, applied, failed = apply_search_replace(
        {"src/main.py": "before\nvalue = 1\nafter\n"}, blocks
    )
    assert files["src/main.py"] == "before\nvalue = 2\nafter\n"
    assert (applied, failed) == (1, 0)


def test_multiple_search_replace_blocks_apply_in_order():
    response = """<<<<<<< SEARCH a.py
one
=======
ONE
>>>>>>> REPLACE
<<<<<<< SEARCH b.py
two
=======
TWO
>>>>>>> REPLACE
"""
    files, applied, failed = apply_search_replace(
        {"a.py": "one\n", "b.py": "two\n"}, parse_search_replace(response)
    )
    assert files == {"a.py": "ONE\n", "b.py": "TWO\n"}
    assert (applied, failed) == (2, 0)


def test_failed_match_is_skipped_and_counted():
    blocks = [
        ("a.py", "missing\n", "replacement\n"),
        ("a.py", "present\n", "changed\n"),
        ("unknown.py", "x\n", "y\n"),
    ]
    files, applied, failed = apply_search_replace({"a.py": "present\n"}, blocks)
    assert files == {"a.py": "changed\n"}
    assert (applied, failed) == (1, 2)


def test_file_blocks_accept_language_and_plain_fences_without_final_newline():
    response = """### FILE: first.py
```python
print("first")
```
### FILE: notes.txt
```
plain
```"""
    assert parse_file_blocks(response) == {
        "first.py": 'print("first")\n',
        "notes.txt": "plain\n",
    }


def test_file_blocks_preserve_nested_shorter_fences():
    response = """### FILE: guide.md
````markdown
# Guide
```python
print("nested")
```
````
"""
    assert parse_file_blocks(response) == {
        "guide.md": '# Guide\n```python\nprint("nested")\n```\n'
    }
