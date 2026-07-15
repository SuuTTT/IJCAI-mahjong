"""Render docs/blog/*.md into arena/static/blog/ (posts + index).

Tiny original markdown subset: #/##, **bold**, *italic*, `code`, tables,
-/1. lists, links. Run at deploy time:  python scripts/render_blog.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "blog"
DST = REPO / "arena" / "static" / "blog"

STYLE = """
:root { color-scheme: dark; }
body { margin:0; background:#0d1117; color:#cdd6e3; font-family:Georgia,serif; }
.wrap { max-width:720px; margin:0 auto; padding:34px 22px 80px; line-height:1.75; font-size:16.5px; }
h1 { font-size:27px; line-height:1.3; color:#eef2f8; } h2 { color:#dfe7f2; margin-top:30px; }
a { color:#6cf; } code { background:#1a2230; padding:1px 6px; border-radius:5px; font-size:14px; }
table { border-collapse:collapse; font-size:14.5px; font-family:system-ui; margin:10px 0; }
th,td { border-bottom:1px solid #2a3547; padding:7px 12px; text-align:left; }
li { margin:8px 0; }
.nav { font-family:system-ui; font-size:13px; margin-bottom:18px; }
.post { display:block; background:#141b26; border:1px solid #223; border-radius:12px;
        padding:16px 20px; margin:14px 0; text-decoration:none; }
.post h3 { margin:0 0 6px; color:#eef2f8; } .post p { margin:0; color:#89a; font-size:14px; }
.date { color:#678; font-size:12.5px; font-family:system-ui; }
"""


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
               r'<img src="\2" alt="\1" style="max-width:100%;border-radius:8px;margin:12px 0">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def render_body(md: str) -> str:
    out, intable, inlist = [], False, False
    for line in md.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "td" if intable else "th"
            if not intable:
                out.append("<table>")
                intable = True
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>"
                                        for c in cells) + "</tr>")
            continue
        if intable:
            out.append("</table>")
            intable = False
        bullet = line.startswith("- ")
        num = re.match(r"^(\d+)\. (.*)$", line)
        if (bullet or num) and not inlist:
            out.append("<ul>" if bullet else "<ol>")
            inlist = "ul" if bullet else "ol"
        if inlist and not (bullet or num) and line.strip():
            out.append(f"</{inlist}>")
            inlist = False
        if bullet:
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if num:
            out.append(f'<li value="{num.group(1)}">{inline(num.group(2))}</li>')
            continue
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if intable:
        out.append("</table>")
    if inlist:
        out.append(f"</{inlist}>")
    return "\n".join(out)


def page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Ludus</title><style>{STYLE}</style></head>
<body><div class="wrap">
<div class="nav"><a href="/">← platform</a> · <a href="/blog">devlog</a> ·
<a href="/league">league</a> · <a href="/play">play</a></div>
{body}
</div></body></html>"""


def render_dir(src: Path, dst: Path, title: str, base: str):
    dst.mkdir(parents=True, exist_ok=True)
    posts = []
    for md_file in sorted(src.glob("*.md"), reverse=True):
        md = md_file.read_text()
        t = next((ln[2:] for ln in md.split("\n") if ln.startswith("# ")),
                 md_file.stem)
        first_p = next((ln for ln in md.split("\n")
                        if ln.strip() and not ln.startswith(("#", "*", "-", "|"))),
                       "")
        slug = md_file.stem
        (dst / f"{slug}.html").write_text(page(t, render_body(md)))
        posts.append((slug, t, first_p))
        print(f"rendered {base}/{slug}")
    cards = "\n".join(
        f'<a class="post" href="/{base}/{slug}">'
        f"<h3>{html.escape(t)}</h3><p>{html.escape(blurb[:180])}</p></a>"
        for slug, t, blurb in posts)
    (dst / "index.html").write_text(page(title, f"<h1>{title}</h1>\n{cards}"))


def main():
    render_dir(REPO / "docs" / "wiki", REPO / "arena" / "static" / "wiki",
               "📖 Ludus Wiki", "wiki")
    DST.mkdir(parents=True, exist_ok=True)
    posts = []
    for md_file in sorted(SRC.glob("*.md"), reverse=True):
        md = md_file.read_text()
        title = next((ln[2:] for ln in md.split("\n") if ln.startswith("# ")),
                     md_file.stem)
        first_p = next((ln for ln in md.split("\n")
                        if ln.strip() and not ln.startswith(("#", "*", "-", "|"))),
                       "")
        slug = md_file.stem
        date = slug[:10]
        (DST / f"{slug}.html").write_text(page(title, render_body(md)))
        posts.append((slug, date, title, first_p))
        print(f"rendered {slug}")
    cards = "\n".join(
        f'<a class="post" href="/blog/{slug}"><span class="date">{date}</span>'
        f"<h3>{html.escape(title)}</h3><p>{html.escape(blurb[:180])}</p></a>"
        for slug, date, title, blurb in posts)
    (DST / "index.html").write_text(page(
        "Devlog", f"<h1>📓 Ludus Devlog</h1>\n{cards}"))
    print(f"index: {len(posts)} posts")


if __name__ == "__main__":
    main()
