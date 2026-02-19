# %%
# Import dependencies and set root path
from src.all_deps import *
ROOT = Path("/Users/marianocrosetti/Desktop/tesina/data/telegramas-urn-sett-2021")
ROOT

# %%
# Find all PDF files in the directory tree
pdfs = sorted(ROOT.rglob("*.pdf"))
pdfs[:5], len(pdfs)

# %%
# Get page counts for all PDFs (using pypdf for speed)
get_pages = lambda p: len(PdfReader(p).pages)
counts = [(p, get_pages(p)) for p in tqdm.tqdm(pdfs)]
counts[:10], len(counts)

# %%
# Filter PDFs with 2 or more pages
multi = [(p, n) for p, n in counts if n >= 2]
len(multi), [(p, n) for p, n in multi]

# %%
