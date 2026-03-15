#!/usr/bin/env python3
"""Generate vbt_sanskrit.md and vbt_english.md from vbt_corpus.py."""

import sys
sys.path.insert(0, "/Users/toeinriver/Projects/sa-embedding")

from vbt_corpus import VBT_CORPUS, VBT_TRANSLATIONS


def write_md(path, title, verses):
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        for i, verse in enumerate(verses):
            f.write(f"## Verse {i}\n\n{verse}\n\n")


write_md("vbt_sanskrit.md", "Vijñāna Bhairava Tantra — Sanskrit", VBT_CORPUS)
write_md("vbt_english.md", "Vijñāna Bhairava Tantra — English", VBT_TRANSLATIONS)

print(f"Wrote vbt_sanskrit.md ({len(VBT_CORPUS)} verses)")
print(f"Wrote vbt_english.md ({len(VBT_TRANSLATIONS)} verses)")
