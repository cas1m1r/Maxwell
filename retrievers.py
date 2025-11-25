# retrievers.py
import requests
from dataclasses import dataclass
from typing import List, Optional
import wikipediaapi
import textwrap

@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None


class WikipediaRetriever:
    """
    Minimal example retriever using Wikipedia's search API.
    This is just one backend – later you can add more.
    """

    API_SEARCH = "https://en.wikipedia.org/w/api.php"
    API_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def search(self, query: str, max_results: int = 5) -> List[WebResult]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }
        resp = requests.get(self.API_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results: List[WebResult] = []

        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            url_title = title.replace(" ", "_")
            url = f"https://en.wikipedia.org/wiki/{url_title}"
            snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            results.append(WebResult(title=title, url=url, snippet=snippet))
            print(f'{textwrap.wrap(snippet)}')
        return results

    def fetch_summary(self, title: str) -> Optional[str]:
        url_title = title.replace(" ", "_")
        url = self.API_SUMMARY + url_title
        resp = requests.get(url, timeout=10)
        print(f'[EXPLORER]::Fetching Summary From >> {url}')
        if not resp.ok:
            return None
        data = resp.json()
        # 'extract' is a plain-text summary
        return data.get("extract")



# retrievers.py
from dataclasses import dataclass
from typing import List, Optional
import requests


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    content: Optional[str] = None


class WikipediaRetriever:
    API_SEARCH = "https://en.wikipedia.org/w/api.php"
    def search(self, query: str, max_results: int = 5) -> str:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }
        print(f'[RETRIEVER]::QUERYING >> {query}')
        wiki = wikipediaapi.Wikipedia(user_agent='Maxwell', language='en')
        page = wiki.page(query.replace(" ", "_"))
        out = ''
        if page.exists():
            print(f'[RETRIEVER]::Page Exists ({len(page.summary)} tokens pulled)')
            out = page.summary
        return out
