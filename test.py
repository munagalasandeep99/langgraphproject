import asyncio
import re
from playwright.async_api import async_playwright

# ── CONFIG ──────────────────────────────────────────────────
URLS_FILE   = "all_chapters.txt"
NOVEL_FILE  = "ancient_godly_monarch.txt"
MIN_CHARS   = 200    # chapters below this are considered missing
WORKERS     = 3      # lower workers for JS-heavy pages
TIMEOUT     = 60000  # ms - increased timeout
# ────────────────────────────────────────────────────────────

# ── STEP 1: Find missing chapter numbers ────────────────────
print("Scanning for missing chapters...")
with open(NOVEL_FILE, "r", encoding="utf-8") as f:
    content = f.read()

chapters_split = re.split(r'\nChapter (\d+)\n={40}\n', content)
missing_nums = set()
i = 1
while i < len(chapters_split) - 1:
    num = int(chapters_split[i])
    text = chapters_split[i+1].strip()
    if len(text) < MIN_CHARS:
        missing_nums.add(num)
    i += 2

print(f"Found {len(missing_nums)} missing chapters\n")

# ── STEP 2: Load URLs for missing chapters only ──────────────
with open(URLS_FILE, "r") as f:
    all_urls = [line.strip() for line in f if line.strip()]

missing_urls = []
for url in all_urls:
    m = re.search(r'/chapter-(\d+)', url)
    if m and int(m.group(1)) in missing_nums:
        missing_urls.append(url)

missing_urls.sort(key=lambda u: int(re.search(r'/chapter-(\d+)', u).group(1)))
print(f"Found {len(missing_urls)} URLs to re-scrape\n")

async def scrape_chapter(context, url, semaphore, total, counter):
    match = re.search(r'/chapter-(\d+)', url)
    chapter_num = int(match.group(1)) if match else 0

    async with semaphore:
        page = await context.new_page()
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="networkidle")

            # Wait until at least one <p> inside #chapter-content has real text (not empty)
            await page.wait_for_function(
                """() => {
                    const ps = document.querySelectorAll('#chapter-content p');
                    return Array.from(ps).some(p => p.innerText.trim().length > 50);
                }""",
                timeout=TIMEOUT
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
                print(f"[{counter[0]}/{total}] ✗ Chapter {chapter_num}: Still locked ({len(text)} chars)")
                return (chapter_num, None)

            print(f"[{counter[0]}/{total}] ✓ Chapter {chapter_num} ({len(text)} chars)")
            return (chapter_num, text)

        except Exception as e:
            counter[0] += 1
            print(f"[{counter[0]}/{total}] ✗ Chapter {chapter_num}: {e}")
            return (chapter_num, None)
        finally:
            await page.close()

async def main():
    total = len(missing_urls)
    counter = [0]
    semaphore = asyncio.Semaphore(WORKERS)
    new_content = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        tasks = [scrape_chapter(context, url, semaphore, total, counter) for url in missing_urls]
        all_results = await asyncio.gather(*tasks)
        await browser.close()

    for chapter_num, text in all_results:
        if text:
            new_content[chapter_num] = text

    print(f"\nSuccessfully scraped {len(new_content)}/{total} chapters.")
    still_missing = missing_nums - set(new_content.keys())
    if still_missing:
        print(f"Still locked/missing ({len(still_missing)}): {sorted(still_missing)[:20]}"
              f"{'...' if len(still_missing) > 20 else ''}")

    # ── STEP 3: Patch novel file ─────────────────────────────
    if not new_content:
        print("Nothing to patch.")
        return

    print("\nPatching novel file...")
    with open(NOVEL_FILE, "r", encoding="utf-8") as f:
        novel = f.read()

    def replace_chapter(match_obj):
        num = int(match_obj.group(1))
        if num in new_content:
            return f"\nChapter {num}\n{'='*40}\n\n{new_content[num]}\n"
        return match_obj.group(0)

    patched = re.sub(
        r'\nChapter (\d+)\n={40}\n.*?(?=\nChapter \d+\n={40}\n|$)',
        replace_chapter,
        novel,
        flags=re.DOTALL
    )

    with open(NOVEL_FILE, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"Done! Patched {len(new_content)} chapters into {NOVEL_FILE}")

asyncio.run(main())