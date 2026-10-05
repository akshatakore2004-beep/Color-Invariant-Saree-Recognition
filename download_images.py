from icrawler.builtin import BingImageCrawler
queries = {"paithani": "paithani saree", "banarasi": "banarasi saree", "kanjivaram": "kanjivaram silk saree"}
for cls, q in queries.items():
    BingImageCrawler(storage={"root_dir": f"data/{cls}"}).crawl(keyword=q, max_num=60)
