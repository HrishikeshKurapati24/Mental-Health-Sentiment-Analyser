import os
import csv
import xml.etree.ElementTree as ET

input_folder = "/Users/hrishikeshkurapati/Downloads/blogs"  # replace with your folder
output_csv = "blog_authorship.csv"

rows = []

# Iterate through each XML file
for filename in os.listdir(input_folder):
    if not filename.endswith(".xml"):
        continue
    
    try:
        # Extract metadata from filename
        parts = filename.split(".")
        blogger_id = parts[0]
        gender = parts[1]
        age = parts[2]
        industry = parts[3]
        sign = parts[4]

        # Parse the XML
        tree = ET.parse(os.path.join(input_folder, filename))
        root = tree.getroot()

        # Extract text content, replace <br /> with spaces
        text = root.text.replace("<br />", " ").replace("\n", " ").strip()

        # Save as a row
        rows.append({
            "id": blogger_id,
            "gender": gender,
            "age": age,
            "industry": industry,
            "sign": sign,
            "text": text
        })

    except Exception as e:
        print(f"Skipping {filename} due to error: {e}")

# Write to CSV
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["id", "gender", "age", "industry", "sign", "text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Done! CSV saved as: {output_csv}")