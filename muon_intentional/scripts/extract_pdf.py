import sys
sys.path.insert(0, "/home/15Cu2Plus/.local/lib/python3.12/site-packages")
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
out = []
for i, p in enumerate(r.pages):
    out.append(f"\n===== PAGE {i+1} =====\n" + (p.extract_text() or ""))
open(sys.argv[2], "w").write("\n".join(out))
print("pages", len(r.pages))
