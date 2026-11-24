"""Convert a Markdown report draft into a formatted HTML document.

The generated HTML follows the local docx conversion rules:
- A4 page size and page furniture (page numbers, no cover furniture).
- Semantic cover, table of contents, and body sections.
- CSS variables for typography, colors, and spacing.
- Markdown tables, fenced code blocks, images, and headings.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import markdown


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)


def build_heading_ids(markdown_text: str) -> tuple[str, dict[str, str]]:
    heading_map: dict[str, str] = {}
    lines = markdown_text.splitlines()
    for line in lines:
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if not match:
            continue
        level, title = match.groups()
        anchor = slugify(title)
        heading_map[title.strip()] = anchor
    return markdown_text, heading_map


def add_heading_anchors(html: str, heading_map: dict[str, str]) -> str:
    for title, anchor in heading_map.items():
        escaped = re.escape(title)
        pattern = re.compile(rf"<h([1-3])>(\s*){escaped}(\s*)</h\1>")

        def replace(match: re.Match[str], title: str = title, anchor: str = anchor) -> str:
            level = match.group(1)
            leading_space = match.group(2)
            trailing_space = match.group(3)
            return f'<h{level} id="{anchor}">{leading_space}{title}{trailing_space}</h{level}>'

        html = pattern.sub(replace, html)
    return html


def transform_images(html: str) -> str:
    pattern = re.compile(r"<p>\s*<img([^>]*)>\s*</p>", re.IGNORECASE)
    attributes = re.compile(r"alt=\"([^\"]*)\"|src=\"([^\"]*)\"")

    def replace(match: re.Match[str]) -> str:
        attrs = match.group(1)
        alt = ""
        src = ""
        for key, value in attributes.findall(attrs):
            if key:
                alt = key
            if value:
                src = value
        caption = alt or "Figure"
        return (
            f"<p class=\"figure-image\"><img src=\"{src}\" alt=\"{alt}\"></p>"
            f"<p class=\"figure-caption\">{caption}</p>"
        )

    return pattern.sub(replace, html)


def build_toc(markdown_text: str, heading_map: dict[str, str]) -> str:
    entries: list[tuple[int, str, str]] = []
    for line in markdown_text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        anchor = heading_map.get(title, slugify(title))
        if level <= 2:
            entries.append((level, title, anchor))
    if not entries:
        return ""
    items: list[str] = []
    for level, title, anchor in entries:
        indent = " class=\"toc-level-2\"" if level == 2 else ""
        items.append(f"<li{indent}><a href=\"#{anchor}\">{title}</a></li>")
    return "<ol class=\"toc-list\">" + "".join(items) + "</ol>"


def convert(config: dict, markdown_path: Path, output_path: Path) -> None:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    markdown_text, heading_map = build_heading_ids(markdown_text)
    body_html = markdown.markdown(
        markdown_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )
    body_html = add_heading_anchors(body_html, heading_map)
    body_html = transform_images(body_html)
    body_html = re.sub(
        r"(<h1[^>]*>Abstract</h1>)(.*?)(<h1)",
        lambda match: f'{match.group(1)}<div class="abstract">{match.group(2)}</div>{match.group(3)}',
        body_html,
        flags=re.DOTALL,
    )
    body_html = re.sub(
        r"(<h1[^>]*>References</h1>)(.*?)(<h1)",
        lambda match: f'{match.group(1)}<div class="references">{match.group(2)}</div>{match.group(3)}',
        body_html,
        flags=re.DOTALL,
    )
    toc_html = build_toc(markdown_text, heading_map)

    title = config["title"]
    subtitle = config.get("subtitle", "")
    metadata_lines = [
        f"Course: {config['course']}" if config.get("course") else "",
        f"Student: {config['student']}" if config.get("student") else "",
        f"Student ID: {config['student_id']}" if config.get("student_id") else "",
        f"Date: {config['date']}" if config.get("date") else "",
        f"GitHub: {config['github_url']}" if config.get("github_url") else "",
    ]
    metadata_html = "<br>".join(line for line in metadata_lines if line)
    toc_block = (
        f"<nav class=\"doc-toc\" aria-label=\"Table of contents\"><p class=\"toc-title\">Table of Contents</p>{toc_html}</nav>"
        if toc_html
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"docx-page-size\" content=\"A4\">
<title>{title}</title>
<style>
:root {{
  --ff-body: "Times New Roman", "SimSun", serif;
  --ff-heading: "Times New Roman", "SimSun", serif;
  --ff-mono: "Consolas", "SimSun", monospace;
  --fs-body: 12pt;
  --fs-small: 10pt;
  --fs-code: 10.5pt;
  --fs-h1: 18pt;
  --fs-h2: 15pt;
  --fs-h3: 13pt;
  --fs-cover-title: 28pt;
  --fs-cover-subtitle: 16pt;
  --fs-cover-meta: 12pt;
  --lh-body: 1.55;
  --lh-heading: 1.25;
  --fw-bold: 700;
  --fw-normal: 400;
  --color-primary: #1F4E79;
  --color-secondary: #2E75B6;
  --color-text: #1A1A1A;
  --color-muted: #595959;
  --color-border: #BFBFBF;
  --color-bg: #FFFFFF;
  --color-panel: #F4F7FA;
  --color-code-bg: #F5F5F5;
  --spacing-paragraph: 0.75em;
  --spacing-section: 1.8em;
  --spacing-block: 1.2em;
  --margin-page: 2.5cm;
  --page-content-width: 16cm;
}}
@page {{
  margin: var(--margin-page);
  @bottom-center {{ content: counter(page); }}
}}
@page cover {{
  @bottom-center {{ content: none; }}
}}
section[role="cover"] {{ page: cover; }}
body {{
  font-family: var(--ff-body);
  font-size: var(--fs-body);
  line-height: var(--lh-body);
  color: var(--color-text);
  background: var(--color-bg);
  max-width: var(--page-content-width);
  margin-left: auto;
  margin-right: auto;
}}
section[role="cover"] {{
  text-align: center;
  padding-top: 5cm;
}}
section[role="cover"] > p {{
  text-align: center;
  text-indent: 0;
}}
.cover-title {{
  font-family: var(--ff-heading);
  font-size: var(--fs-cover-title);
  font-weight: var(--fw-bold);
  color: var(--color-primary);
  margin-bottom: var(--spacing-section);
  text-indent: 0;
}}
.cover-subtitle {{
  font-family: var(--ff-heading);
  font-size: var(--fs-cover-subtitle);
  color: var(--color-secondary);
  margin-bottom: var(--spacing-section);
  text-indent: 0;
}}
.cover-meta {{
  font-family: var(--ff-body);
  font-size: var(--fs-cover-meta);
  color: var(--color-muted);
  line-height: var(--lh-body);
  text-indent: 0;
}}
.toc-title {{
  font-family: var(--ff-heading);
  font-size: var(--fs-h1);
  font-weight: var(--fw-bold);
  color: var(--color-primary);
  margin-bottom: var(--spacing-section);
  text-align: center;
  text-indent: 0;
}}
.toc-list, .toc-list ol, .toc-list ul {{
  list-style: none;
  padding-left: 0;
  margin-left: 0;
}}
.toc-list li {{
  margin-bottom: 0.35em;
}}
.toc-level-2 {{
  margin-left: 1.5em;
}}
.toc-list a {{
  color: var(--color-text);
  text-decoration: none;
}}
h1, h2, h3, h4, h5, h6 {{
  font-family: var(--ff-heading);
  color: var(--color-primary);
  line-height: var(--lh-heading);
  margin-top: var(--spacing-section);
  margin-bottom: var(--spacing-block);
  text-indent: 0;
}}
h1 {{ font-size: var(--fs-h1); font-weight: var(--fw-bold); }}
h2 {{ font-size: var(--fs-h2); font-weight: var(--fw-bold); color: var(--color-secondary); }}
h3 {{ font-size: var(--fs-h3); font-weight: var(--fw-bold); color: var(--color-secondary); }}
.abstract, .references {{
  margin-top: var(--spacing-section);
  margin-bottom: var(--spacing-section);
}}
p {{
  margin-top: var(--spacing-paragraph);
  margin-bottom: var(--spacing-paragraph);
  text-align: justify;
  text-indent: 2em;
}}
ul, ol {{
  margin-top: var(--spacing-paragraph);
  margin-bottom: var(--spacing-paragraph);
  padding-left: 2em;
}}
li {{
  margin-bottom: 0.35em;
  text-align: justify;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: var(--spacing-block);
  margin-bottom: var(--spacing-block);
  page-break-inside: avoid;
}}
thead {{
  background: var(--color-panel);
  color: var(--color-primary);
}}
th, td {{
  border: 1px solid var(--color-border);
  padding: 0.45em 0.6em;
  text-align: left;
  vertical-align: top;
  font-size: var(--fs-small);
}}
th {{
  font-weight: var(--fw-bold);
}}
code {{
  font-family: var(--ff-mono);
  background: var(--color-code-bg);
  padding: 0.1em 0.25em;
  font-size: var(--fs-code);
}}
pre {{
  background: var(--color-code-bg);
  border: 1px solid var(--color-border);
  padding: 0.8em;
  overflow-x: auto;
  page-break-inside: avoid;
}}
pre code {{
  background: none;
  padding: 0;
}}
blockquote {{
  border-left: 3px solid var(--color-secondary);
  background: var(--color-panel);
  padding: 0.7em 1em;
  margin: var(--spacing-block) 0;
  text-indent: 0;
}}
blockquote p {{
  text-indent: 0;
  margin: 0;
}}
.figure-image, .figure-caption {{
  text-align: center;
  text-indent: 0;
  margin-top: var(--spacing-block);
  margin-bottom: 0.25em;
}}
.figure-image img {{
  max-width: 100%;
  height: auto;
}}
.figure-caption {{
  font-size: var(--fs-small);
  color: var(--color-muted);
  margin-bottom: var(--spacing-block);
}}
a {{
  color: var(--color-secondary);
}}
</style>
</head>
<body>
<section role=\"cover\">
  <p class=\"cover-title\">{title}</p>
  <p class=\"cover-subtitle\">{subtitle}</p>
  <p class=\"cover-meta\">{metadata_html}</p>
</section>
<section role=\"cover\">
{toc_block}
</section>
<section role=\"body\" data-page-restart=\"1\">
{body_html}
</section>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    convert(config, args.markdown, args.output)


if __name__ == "__main__":
    main()
