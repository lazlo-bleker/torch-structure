"""Generate the code reference pages, their navigation and an overview page."""

from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

root = Path(__file__).parent.parent
src = root / "src"

nav = mkdocs_gen_files.Nav()
modules_by_package = defaultdict(list)

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__" or parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"# {parts[-1]}\n", file=fd)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    # list modules without the torch_structure prefix, grouped by subpackage
    nav[parts[1:]] = doc_path.as_posix()
    package = parts[1] if len(parts) > 2 else ""
    modules_by_package[package].append((".".join(parts), doc_path.as_posix()))

with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    print("# API Reference\n", file=fd)
    for package in sorted(modules_by_package):
        print(f"## {package or 'torch_structure'}\n", file=fd)
        for identifier, doc_path in modules_by_package[package]:
            print(f"- [`{identifier}`]({doc_path})", file=fd)
        print("", file=fd)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    # the overview first, so it becomes the landing page of the section (navigation.indexes)
    nav_file.write("* [API Reference](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
