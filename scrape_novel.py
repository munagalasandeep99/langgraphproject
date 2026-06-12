"""
Build ancient_godly_monarch.txt from scratch using the URLs in all_chapters.txt.


Usage:
    python scrape_novel.py              # scrape ALL chapters (slow: 2052 of them)
    python scrape_novel.py --limit 50   # scrape only the first 50 (good for testing)

It is resumable: chapters already present (with enough text) in the output file
are skipped, so you can stop with Ctrl-C and rerun to continue.

Requires Playwright:
    pip install playwright
    playwright install chromium
"""

import argparse
import asyncio
import re
from pathlib import Path

from playwright.async_api import async_playwright

URLS_FILE = "all_chapters.txt"
NOVEL_FILE = "ancient_godly_monarch.txt"
MIN_CHARS = 200      # below this, a chapter is considered locked/empty
WORKERS = 3          # be polite to the site; JS-heavy pages
TIMEOUT = 60000      # ms
CHAPTER_HEADER = "=" * 40

CHAPTER_NUM_RE = re.compile(r"/chapter-(\d+)")
EXISTING_RE = re.compile(r"\nChapter (\d+)\n={40}\n(.*?)(?=\nChapter \d+\n={40}\n|$)", re.DOTALL)


def load_existing(path: Path) -> dict[int, str]:
    """Read chapters already scraped so we can resume without re-fetching them."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    found = {}
    for m in EXISTING_RE.finditer(text):
        num = int(m.group(1))
        body = m.group(2).strip()
        if len(body) >= MIN_CHARS:
            found[num] = body
    return found


def chapter_num(url: str) -> int:
    m = CHAPTER_NUM_RE.search(url)
    return int(m.group(1)) if m else 0


async def scrape_chapter(context, url, semaphore, total, counter):
    num = chapter_num(url)
    async with semaphore:
        page = await context.new_page()
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="networkidle")
            await page.wait_for_function(
                """() => {
                    const ps = document.querySelectorAll('#chapter-content p');
                    return Array.from(ps).some(p => p.innerText.trim().length > 50);
                }""",
                timeout=TIMEOUT,
            )
            paragraphs = await page.query_selector_all("#chapter-content p")
            texts = []
            for p in paragraphs:
                t = (await p.inner_text()).strip()
                if t and len(t) > 10:
                    texts.append(t)
            text = "\n\n".join(texts)

            counter[0] += 1
            if len(text) < MIN_CHARS:
                print(f"[{counter[0]}/{total}] x Chapter {num}: locked/empty ({len(text)} chars)")
                return (num, None)
            print(f"[{counter[0]}/{total}] ok Chapter {num} ({len(text)} chars)")
            return (num, text)
        except Exception as e:
            counter[0] += 1
            print(f"[{counter[0]}/{total}] x Chapter {num}: {e}")
            return (num, None)
        finally:
            await page.close()


def write_novel(path: Path, chapters: dict[int, str]) -> None:
    """Write all chapters in ascending order in index.py's expected format."""
    with path.open("w", encoding="utf-8") as f:
        for num in sorted(chapters):
            f.write(f"\nChapter {num}\n{CHAPTER_HEADER}\n\n{chapters[num]}\n")


async def main(limit: int | None) -> None:
    root = Path(__file__).parent
    urls_path = root / URLS_FILE
    novel_path = root / NOVEL_FILE

    if not urls_path.exists():
        print(f"Error: {urls_path} not found.")
        return

    all_urls = [line.strip() for line in urls_path.read_text().splitlines() if line.strip()]
    all_urls.sort(key=chapter_num)
    if limit:
        all_urls = all_urls[:limit]

    existing = load_existing(novel_path)
    if existing:
        print(f"Resuming: {len(existing)} chapters already scraped, will skip those.")

    todo = [u for u in all_urls if chapter_num(u) not in existing]
    print(f"{len(todo)} chapters to scrape (of {len(all_urls)} targeted).\n")
    if not todo:
        print("Nothing to do — already complete for this range.")
        return

    chapters = dict(existing)
    counter = [0]
    semaphore = asyncio.Semaphore(WORKERS)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Process in batches so we can save progress periodically (resumable).
        BATCH = 30
        for i in range(0, len(todo), BATCH):
            batch_urls = todo[i : i + BATCH]
            results = await asyncio.gather(
                *[scrape_chapter(context, u, semaphore, len(todo), counter) for u in batch_urls]
            )
            for num, text in results:
                if text:
                    chapters[num] = text
            write_novel(novel_path, chapters)  # checkpoint after each batch
            print(f"  ...saved progress ({len(chapters)} chapters so far)\n")

        await browser.close()

    print(f"\nDone. {len(chapters)} chapters written to {NOVEL_FILE}.")
    print("Next: python index.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scrape the first N chapters (for testing).")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
