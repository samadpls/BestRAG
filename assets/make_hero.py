"""Write assets/hero.svg, the animated README banner.

The text is drawn as outlines because GitHub shows README images through a proxy that loads no web
fonts. Fonts (from github.com/google/fonts) go in data/fonts: Bricolage.ttf and Geist.ttf.

    uv run --no-project --with uharfbuzz --with fonttools python assets/make_hero.py
"""

import math
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONTS = Path("data/fonts")
NAVY, VIOLET, CRIMSON, TILE, MUTED, RULE = "#24376C", "#7E85B6", "#D22650", "#E3ECFB", "#4A5068", "#BCC6DD"
CARD, BORDER, GLOW = "#F7FAFF", "#DCE4F3", "#E9F0FC"
VIOLET_INK = "#575E96"  # the violet is too pale for text on the card
W, H, LOOP = 1280, 400, 10
MEANING, WORDS, MERGED = (0, 30), (31, 62), (63, 98)  # percent of the loop
# The two curves of logo.svg, in its 512-unit box; the logo is the finished state.
SOLID = ((100, 150), (200, 150), (236, 256), (326, 256))
DOTTED = ((100, 362), (200, 362), (236, 256), (326, 256))


def num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


class Font:
    def __init__(self, file, **axes):
        face = hb.Face(hb.Blob.from_file_path(FONTS / file))
        self.font, self.unit = hb.Font(face), 1 / face.upem
        self.font.set_variations(axes)

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.font, buf, {})
        return zip(buf.glyph_infos, buf.glyph_positions)

    def width(self, text, size, tracking=0):
        return sum(p.x_advance * size * self.unit + tracking for _, p in self._shape(text))

    def path(self, text, x, y, size, tracking=0):
        s = size * self.unit
        pen = SVGPathPen(None, ntos=num)
        for info, p in self._shape(text):
            self.font.draw_glyph_with_pen(
                info.codepoint, TransformPen(pen, (s, 0, 0, -s, x + p.x_offset * s, y - p.y_offset * s))
            )
            x += p.x_advance * s + tracking
        return pen.getCommands()


def curve(p):
    return "M{} {}C{} {} {} {} {} {}".format(*(c for xy in p for c in xy))


def dash_dots(p, step=54, n=5000):
    """Centres of the round dots that stroke-dasharray="0 54" draws along the cubic p."""
    dots, run, prev = [p[0]], 0.0, p[0]
    for k in range(1, n + 1):
        u = k / n
        cur = [(1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * b + 3 * (1 - u) * u * u * c + u**3 * d for a, b, c, d in zip(*p)]
        run += math.dist(prev, cur)
        if run >= step * len(dots):
            dots.append(cur)
        prev = cur
    return dots


def fade(name, windows, base):
    """Opacity 1 inside each (start, end) percent window, with 1% fades."""
    stops = {0.0: 0.0, 100.0: 0.0}
    for a, b in windows:
        stops |= {max(a - 1, 0): 0.0, a: 1.0, b - 1: 1.0, b: 0.0}
    if windows[0][0] == 0:
        stops[0.0] = 1.0
    body = "".join(f"{num(t)}%{{opacity:{o:g}}}" for t, o in sorted(stops.items()))
    return f"@keyframes {name}{{{body}}}.{name}{{opacity:{base};animation:{name} {LOOP}s linear infinite}}"


def lights(name, start):
    """Dim until start percent, then a small pop to full, lit until the loop resets."""
    return (
        f"@keyframes {name}{{0%,{start}%{{opacity:.25;transform:scale(1)}}{start + 2}%{{opacity:1;transform:scale(1.3)}}"
        f"{start + 5}%,97%{{opacity:1;transform:scale(1)}}100%{{opacity:.25;transform:scale(1)}}}}"
        f".{name}{{transform-box:fill-box;transform-origin:center;animation:{name} {LOOP}s ease-out infinite}}"
    )


def main():
    title = Font("Bricolage.ttf", wght=800, opsz=96)
    text = Font("Geist.ttf", wght=400)
    bold = Font("Geist.ttf", wght=600)

    x0 = 432
    title_w = title.width("BestRAG", 112, -3)
    rag_x = x0 + title_w - title.width("RAG", 112, -3)  # keeps the t-R kerning of the whole word
    tagline = "Hybrid search for RAG on Postgres and Qdrant"
    parts = [
        f'<path fill="{NAVY}" d="{title.path("Best", x0, 158, 112, -3)}"/>',
        f'<path fill="{CRIMSON}" d="{title.path("RAG", rag_x, 158, 112, -3)}"/>',
        f'<path fill="{MUTED}" d="{text.path(tagline, x0, 216, 30)}"/>',
    ]
    status = [
        ("meaning", NAVY, NAVY, "Searching by meaning", MEANING, 0),
        ("words", VIOLET, VIOLET_INK, "Searching by exact words", WORDS, 0),
        ("merged", CRIMSON, CRIMSON, "Merged: best passages first", MERGED, 1),
    ]
    css = []
    for name, color, ink, words, window, base in status:
        parts.append(
            f'<g class="{name}"><circle cx="{x0 + 8}" cy="271" r="7" fill="{color}"/>'
            f'<path fill="{ink}" d="{bold.path(words, x0 + 26, 281, 28)}"/></g>'
        )
        css.append(fade(name, [window], base))

    links = ["github.com/samadpls", "samadpls.com", "Open source, MIT"]
    x = x0
    for k, link in enumerate(links):
        if k:
            parts.append(f'<circle cx="{num(x + 14)}" cy="327" r="2.5" fill="{RULE}"/>')
            x += 28
        parts.append(f'<path fill="{NAVY}" d="{bold.path(link, x, 334, 21)}"/>')
        x += bold.width(link, 21)
    assert max(x, x0 + title_w, x0 + text.width(tagline, 30)) < W - 60

    dots = dash_dots(DOTTED)
    assert len(dots) == 5, dots  # the five dots logo.svg shows
    step = (WORDS[1] - WORDS[0] - 9) // len(dots)
    circles = "".join(
        f'<circle class="d{i}" cx="{num(cx)}" cy="{num(cy)}" r="21"/>' for i, (cx, cy) in enumerate(dots)
    )
    css += [lights(f"d{i}", WORDS[0] + 2 + i * step) for i in range(len(dots))]
    css.append(
        "@keyframes draw{0%,2%{stroke-dashoffset:1.02}24%,97%{stroke-dashoffset:0;opacity:1}100%{stroke-dashoffset:0;opacity:0}}"
        f".draw{{stroke-dasharray:1 2;animation:draw {LOOP}s ease-in-out infinite}}"
    )
    css.append(
        f"@keyframes pop{{0%,{MERGED[0]}%{{transform:scale(0)}}{MERGED[0] + 4}%{{transform:scale(1.15)}}"
        f"{MERGED[0] + 7}%{{transform:scale(.96)}}{MERGED[0] + 10}%,97%{{transform:scale(1)}}100%{{transform:scale(0)}}}}"
        f".pop{{transform-box:fill-box;transform-origin:center;animation:pop {LOOP}s ease-out infinite}}"
    )
    css.append("@media (prefers-reduced-motion:reduce){*{animation:none!important}}")

    # baari's tile: 116 of its 120 units, drawn 190 px wide at (140, 105)
    tile = 116 * 190 / 120
    label = "BestRAG: hybrid search for RAG on Postgres and Qdrant"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{label}">
<title>{label}</title>
<defs><radialGradient id="glow" cx="235" cy="200" r="430" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{GLOW}"/><stop offset="1" stop-color="{GLOW}" stop-opacity="0"/></radialGradient></defs>
<style>{"".join(css)}</style>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="24" fill="{CARD}" stroke="{BORDER}"/>
<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="24" fill="url(#glow)"/>
<g transform="translate({140 + 2 * 190 / 120:.2f} {105 + 2 * 190 / 120:.2f}) scale({tile / 512:.4f})">
<rect width="512" height="512" rx="112" fill="{TILE}"/>
<g fill="none" stroke="{NAVY}" stroke-width="42" stroke-linecap="round"><path d="{curve(SOLID)}" opacity=".2"/><path class="draw" d="{curve(SOLID)}" pathLength="1"/></g>
<g fill="{VIOLET}">{circles}</g>
<circle cx="378" cy="256" r="64" fill="{CRIMSON}" opacity=".2"/>
<circle class="pop" cx="378" cy="256" r="64" fill="{CRIMSON}"/>
</g>
{"".join(parts)}
</svg>
"""
    out = Path("assets/hero.svg")
    out.write_text(svg)
    print(out, f"{len(svg) // 1024} KB")


if __name__ == "__main__":
    main()
