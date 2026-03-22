import re, json
from pathlib import Path

def clean_str(s):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(s)).strip()

p = Path("data/processed/clrs_concepts.json")
if not p.exists():
    print("clrs_concepts.json not found")
    exit()

with open(p, encoding="utf-8") as f:
    data = json.load(f)

fixed = 0
for ex in data:
    for c in ex.get("concepts", []):
        for field in ["name", "definition"]:
            orig = c.get(field, "")
            cleaned = clean_str(orig)
            if cleaned != orig:
                c[field] = cleaned
                fixed += 1
        for field in ["prerequisites", "enables"]:
            new_list = []
            for item in c.get(field, []):
                cleaned = clean_str(item)
                if cleaned != item:
                    fixed += 1
                new_list.append(cleaned)
            c[field] = [x for x in new_list if x]

with open(p, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Cleaned {fixed} strings in clrs_concepts.json")