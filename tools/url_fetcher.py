import typer
import requests
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from pathlib import Path
import hashlib
import logging
import json
from datetime import datetime
from typing import Optional
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import gzip
import base64
import json

app = typer.Typer()


def compress_and_encode_html(html_content: str):
    compressed = gzip.compress(html_content.encode("utf-8"))

    base64_encoded = base64.b64encode(compressed).decode("utf-8")

    return base64_encoded


def setup_logging(log_level: str):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def remove_utm_params(url):
    # Parse the URL
    parsed = urlparse(url)

    # Get query parameters as dictionary
    query_dict = parse_qs(parsed.query)

    # Remove all utm_* parameters
    clean_query = {
        key: value for key, value in query_dict.items() if not key.startswith("utm_")
    }

    # Reconstruct URL without UTM parameters
    clean_parsed = parsed._replace(query=urlencode(clean_query, doseq=True))
    return clean_parsed


@app.command()
def fetch_and_store(
    url_file: Path = typer.Argument(..., help="Path to file containing URLs"),
    output_dir: Path = typer.Option(
        "url_storage", help="Base directory for storing data"
    ),
    timeout: int = typer.Option(10, help="Timeout for URL requests in seconds"),
    log_level: str = typer.Option(
        "INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    ),
):
    """Fetch URLs content and store it with metadata in JSON format."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    logger.info(f"Creating base directory: {output_dir}")
    output_dir.mkdir(exist_ok=True)

    def generate_hash(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    try:
        with open(url_file, "r") as f:
            urls = set(f.readlines())
            logger.info(f"Found {len(urls)} URLs in {url_file}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {url_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        raise typer.Exit(code=1)

    processed_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Processing URLs...", total=len(urls))

        for url in urls:
            url = url.strip()
            if not url:
                progress.update(task, advance=1)
                continue

            try:
                parsed_url = remove_utm_params(url)

                domain = parsed_url.netloc

                if not domain:
                    logger.warning(f"Invalid URL (no domain): {url}")
                    error_count += 1
                    progress.update(task, advance=1)
                    continue

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(
                    f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}",
                    params=parsed_url.params,
                    headers=headers,
                )
                response.raise_for_status()
                domain_dir = output_dir / domain
                domain_dir.mkdir(exist_ok=True)

                # Fetch URL content
                url_hash = generate_hash(url)

                content = response.text

                # Store metadata and content
                url_data = {
                    "metadata": {
                        "url": url,
                        "hash": url_hash,
                        "domain": domain,
                        "timestamp": datetime.now().isoformat(),
                        "path": parsed_url.path,
                        "query": parsed_url.query,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type"),
                        "encoding": response.encoding,
                    },
                    "content_encoding": "application/gzip",
                    "base64": True,
                    "content": compress_and_encode_html(content),
                }

                json_file = domain_dir / f"{url_hash}.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(url_data, f, indent=2)

                logger.debug(f"Stored URL data: {url} as {url_hash}.json in {domain}")
                processed_count += 1

            except requests.RequestException as e:
                logger.error(f"Failed to fetch URL {url}: {str(e)}")
                error_count += 1
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                error_count += 1
            finally:
                progress.update(task, advance=1)

    logger.info(
        f"Processing complete. Successfully processed: {processed_count}, Errors: {error_count}"
    )


if __name__ == "__main__":
    app()
