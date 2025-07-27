"""Generate API reference pages."""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src_path = Path("src")

for path in sorted(src_path.rglob("*.py")):
    module_path = path.relative_to(src_path).with_suffix("")
    doc_path = path.relative_to(src_path).with_suffix(".md")
    full_doc_path = Path("reference", "api", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {ident}\n\n")
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())