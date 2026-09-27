"""Fold the payload and the app into one self-contained page.

Run from ui/atlas/ after build_payload.py:  python assemble.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
shell = (HERE / "atlas_shell.html").read_text(encoding="utf-8")
payload = (HERE / "atlas_payload.js").read_text(encoding="utf-8").replace("</", "<\\/")
app = (HERE / "atlas_app.js").read_text(encoding="utf-8")

html = shell.replace("<!--DATA-->", f"<script>{payload}</script>").replace(
    "<!--APP-->", f"<script>\n(() => {{\n{app}\n}})();\n</script>")
out = HERE / "groundwater-atlas.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out.name} {out.stat().st_size / 1e6:.2f} MB")
