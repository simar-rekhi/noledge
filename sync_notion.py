import os
import requests

# Get variables from GitHub Secrets
notion_token = os.getenv("notion_token")
PAGE_ID = "2e765a5c0c2f80159213d93a58a276b7"   # extract code from copy link

def fetch_notion_page():
    url = f"https://api.notion.com/v1/blocks/{PAGE_ID}/children"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Simple logic to extract text from blocks
    markdown_content = ""
    for block in data.get("results", []):
        block_type = block.get("type")
        if block_type == "paragraph":
            rich_text = block["paragraph"]["rich_text"]
            if rich_text:
                markdown_content += rich_text[0]["plain_text"] + "\n\n"
        elif block_type == "heading_1":
            markdown_content += "# " + block["heading_1"]["rich_text"][0]["plain_text"] + "\n\n"
            
    with open("notion_content.md", "w") as f:
        f.write(markdown_content)

if __name__ == "__main__":
    fetch_notion_page()
