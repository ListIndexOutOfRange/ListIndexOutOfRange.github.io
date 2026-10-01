"""Restrict the generated CNPHGE site to its cover until publication is enabled."""
import argparse
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination: Path, mode: str) -> None:
    if mode not in {"title", "full"}:
        raise ValueError("CNPHGE_PUBLICATION must be 'title' or 'full'")
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents and destination != ROOT / "public":
        raise ValueError("Use the generated public directory or an output outside the repository")
    if mode == "full":
        return
    source = ROOT / "static/cnphge"
    html = (source / "index.html").read_text()
    cover = re.search(r'<section class="cover"[^>]*>.*?</section>', html, re.S)
    if cover is None:
        raise ValueError("Cannot find the title slide; refusing to publish the full deck")
    head = html.split("<body>", 1)[0]
    title = re.sub(r'<aside\b.*?</aside>', '', cover.group(), flags=re.S)
    output = head + '<body><div class="reveal"><div class="slides">' + title
    output += '''</div></div><script src="../camma/_reveal/dist/reveal.js"></script>
<script>Reveal.initialize({width:1280,height:720,margin:0.055,center:false,
controls:false,progress:false,slideNumber:false,hash:false,keyboard:false,
touch:false,overview:false,help:false});</script></body></html>'''
    target = destination / "cnphge"
    if not target.is_dir():
        raise ValueError("Build the site before preparing publication")
    # Remove only generated presentation assets, never the working deck.
    shutil.rmtree(target)
    (target / "css").mkdir(parents=True)
    (target / "index.html").write_text(output)
    shutil.copyfile(source / "css/deck.css", target / "css/deck.css")
    print("CNPHGE: published title slide only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mode", choices=["title", "full"], required=True)
    args = parser.parse_args()
    prepare(args.destination, args.mode)
