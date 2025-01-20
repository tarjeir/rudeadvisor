import logging
import glob
import re
import requests
import time
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename="url_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def extract_urls_from_log():
    # The pattern you specified
    url_pattern = (
        r"https://news\.dataelixir\.com/t/t-l-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-z]/"
    )
    found_urls = set()  # Using set to avoid duplicates

    try:
        with open("url_processing.log", "r") as log_file:
            content = log_file.read()
            urls = re.findall(url_pattern, content)
            found_urls.update(urls)
            return found_urls
    except FileNotFoundError:
        print("Log file not found")
        return set()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise Exception()


def find_urls_in_emails(urls_found: set[str]):
    # Firefox-specific headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
    }

    url_pattern = (
        r"https://news\.dataelixir\.com/t/t-l-[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-z]/"
    )
    root_dir = Path("/Users/tarjeiromtveit/projects/private/eml_training/Training Data")
    eml_files = glob.glob(
        "*.eml",
        root_dir=root_dir,
    )
    found_urls = []

    for eml_file in eml_files:
        try:
            with open(root_dir / eml_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                urls = re.findall(url_pattern, content)
                only_urls_that_are_not_found = set(urls) - urls_found
                found_urls.extend(only_urls_that_are_not_found)
        except Exception as e:
            logging.error(f"Error reading file {eml_file}: {str(e)}")

    # Open a file to store only redirected URLs
    with open("redirected_urls.txt", "w") as url_file:
        for url in found_urls:
            try:
                response = requests.get(url, headers=headers, allow_redirects=False)
                logging.info(
                    f"Processed URL: {url} with status code {response.status_code}"
                )
                if "Location" in response.headers:
                    redirected_url = response.headers["Location"]
                    url_file.write(f"{redirected_url}\n")
                    logging.info(f"Redirected URL: {redirected_url}")
                # Add a random pause between 1 and 5 seconds
                time.sleep(random.uniform(1, 5))
            except Exception as e:
                logging.error(f"Error processing URL {url}: {str(e)}")


if __name__ == "__main__":
    urls_found = extract_urls_from_log()
    find_urls_in_emails(urls_found)
