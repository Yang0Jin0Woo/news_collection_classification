"""Conservative, keyword-independent extraction of the current article only.

Static Arc story data follows https://github.com/washingtonpost/ans-schema.
JavaScript is never executed and related_content is never traversed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.parse import parse_qsl, urljoin, urlsplit

from bs4 import BeautifulSoup

from news_classifier.utils.text import clean_text, normalize_key


BODY_SELECTOR = (
    '[itemprop~="articleBody"], #article-view-content-div, #articleBody, '
    '#article-body, #newsBody, #news-body, .article-body, .article_body, '
    '.article-content, .article_content, .news_view, .news-body, .news_body, .view_body'
)
NOISE_TOKEN = re.compile(
    r"(?:^|[-_\s])(?:author|writer|copyright|related|recommend(?:ed|ation)?|"
    r"comments?|advert(?:isement|ising)?|ads?|adbox|banner|subscribe|subscription|paywall|captcha|login|consent|poll|"
    r"fontsize|logo|card|caption|share|sns)(?:$|[-_\s])", re.I,
)
ARTICLE_TYPES = {"NewsArticle", "Article", "ReportageNewsArticle"}
MAX_PAGE_BYTES = 4_000_000


@dataclass(frozen=True)
class BodyExtraction:
    content: str = ""
    method: str = ""
    reason: str = "no_usable_body"
    paragraph_count: int = 0
    candidate_count: int = 0

    def diagnostics(self):
        return {"method": self.method, "reason": self.reason,
                "paragraph_count": self.paragraph_count, "candidate_count": self.candidate_count}


def _remove_noise(soup):
    for tag in list(soup.find_all(True)):
        if tag.parent is None or tag.attrs is None:
            continue
        identity = " ".join([str(tag.get("id", "")), " ".join(tag.get("class", []))])
        if (tag.name in {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "figure", "figcaption", "button"}
                or tag.get("hidden") is not None or tag.get("aria-hidden") == "true"
                or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", tag.get("style", ""), re.I)
                or NOISE_TOKEN.search(identity)):
            tag.decompose()


def _text_blocks(root, title, *, paragraphs_only=False):
    # Preserve spaces inside inline formatting; use line breaks only at block boundaries.
    fragment = BeautifulSoup(str(root), "html.parser")
    if paragraphs_only:
        texts = [p.get_text(" ") for p in fragment.find_all("p")]
    else:
        for tag in fragment.find_all("br"):
            tag.replace_with("\n")
        for tag in fragment.find_all(["p", "div", "section", "li", "blockquote", "h1", "h2", "h3"]):
            tag.insert_before("\n")
            tag.insert_after("\n")
        texts = fragment.get_text(" ").splitlines()
    seen, blocks = {normalize_key(title)}, []
    for value in texts:
        text = clean_text(value)
        key = normalize_key(text)
        if len(text) >= 20 and key not in seen:
            blocks.append(text)
            seen.add(key)
    return " ".join(blocks), len(blocks)


def _same_article_url(value, page_url):
    if not isinstance(value, str) or not value or not page_url:
        return False
    def identity(url):
        p = urlsplit(url)
        return ((p.hostname or "").lower().removeprefix("www."), p.path.rstrip("/"),
                sorted((key, val) for key, val in parse_qsl(p.query)
                       if not key.startswith("utm_") and key not in {"fbclid", "gclid"}))
    try:
        return identity(urljoin(page_url, value)) == identity(page_url)
    except ValueError:
        return False


def _json_ld_bodies(soup, page_url):
    bodies = []
    def visit(value, depth=0):
        if depth > 16:
            return
        if isinstance(value, list):
            for child in value:
                visit(child, depth + 1)
        elif isinstance(value, dict):
            types = value.get("@type", [])
            types = [types] if isinstance(types, str) else types
            body = value.get("articleBody")
            identity = value.get("url") or value.get("mainEntityOfPage") or value.get("@id")
            if isinstance(identity, dict):
                identity = identity.get("@id") or identity.get("url")
            if (isinstance(types, list) and any(isinstance(t, str) and t in ARTICLE_TYPES for t in types)
                    and isinstance(body, str) and (not identity or _same_article_url(identity, page_url))):
                fragment = BeautifulSoup(body, "html.parser")
                _remove_noise(fragment)
                text = clean_text(fragment.get_text(" "))
                if len(text) >= 40:
                    bodies.append(text)
            if "@graph" in value:
                visit(value["@graph"], depth + 1)
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            visit(json.loads(tag.string or tag.get_text()))
        except (ValueError, TypeError, RecursionError):
            continue
    return list(dict.fromkeys(bodies))


def _arc_bodies(soup, page_url, title):
    bodies, restricted = [], False
    for tag in soup.find_all("script"):
        text = tag.string or tag.get_text()
        for match in re.finditer(r"(?<![\w.])(?:window\.)?Fusion\.globalContent\s*=\s*", text):
            try:
                story, _ = json.JSONDecoder().raw_decode(text[match.end():].lstrip())
            except (ValueError, TypeError, RecursionError):
                continue
            if not isinstance(story, dict) or story.get("type") != "story":
                continue
            url = story.get("website_url") or story.get("canonical_url")
            headline = story.get("headlines", {})
            if not isinstance(headline, dict):
                continue
            headline = headline.get("basic")
            if (not _same_article_url(url, page_url) or not isinstance(headline, str)
                    or normalize_key(headline) != normalize_key(title)):
                continue
            restriction = story.get("content_restrictions", {})
            if (not isinstance(restriction, dict)
                    or restriction.get("content_code") not in (None, "", "0", 0)
                    or restriction.get("restricted") is True):
                restricted = True
                continue
            elements = story.get("content_elements")
            if not isinstance(elements, list) or len(elements) > 2000:
                continue
            html = "".join("<p>" + entry["content"] + "</p>" for entry in elements
                           if isinstance(entry, dict) and entry.get("type") == "text"
                           and isinstance(entry.get("content"), str))
            fragment = BeautifulSoup(html, "html.parser")
            _remove_noise(fragment)
            content, count = _text_blocks(fragment, title, paragraphs_only=True)
            if len(content) >= 40:
                bodies.append((content, count))
    return list(dict.fromkeys(bodies)), restricted


def extract_article_text(html, *, title="", page_url=""):
    soup = BeautifulSoup(html[:MAX_PAGE_BYTES], "html.parser")
    if soup.find("rss") is not None or soup.find("feed") is not None:
        return BodyExtraction(reason="rss_document")
    page_title = clean_text(soup.title.get_text(" ") if soup.title else "").casefold()
    if (page_title in {"access denied", "just a moment...", "attention required! | cloudflare"}
            or page_title.startswith("before you continue to google")):
        return BodyExtraction(reason="consent_or_block_page")
    json_bodies = _json_ld_bodies(soup, page_url)
    arc_bodies, restricted = _arc_bodies(soup, page_url, title)
    _remove_noise(soup)
    scoped = []
    for node in soup.select(BODY_SELECTOR):
        content, count = _text_blocks(node, title)
        links = sum(len(a.get_text(" ")) for a in node.find_all("a"))
        if len(content) >= 40 and links / max(1, len(content)) < .35:
            scoped.append((node, content, count))
    # Choose the inner body, not its wrapper with recommendations/author metadata.
    inner = [entry for entry in scoped if not any(
        entry[0] is not other[0] and any(parent is entry[0] for parent in other[0].parents)
        for other in scoped)]
    unique = {normalize_key(text): (text, count) for _, text, count in inner}
    if len(unique) == 1:
        content, count = next(iter(unique.values()))
        return BodyExtraction(content, "BODY_CONTAINER_BLOCKS", "body_found", count, len(inner))
    if len(unique) > 1:
        return BodyExtraction(reason="multiple_body_candidates", candidate_count=len(unique))
    if len(json_bodies) == 1:
        return BodyExtraction(json_bodies[0], "JSON_LD_BODY", "body_found", 1, 1)
    if len(arc_bodies) == 1:
        content, count = arc_bodies[0]
        return BodyExtraction(content, "ARC_STORY_JSON", "body_found", count, 1)
    if len(json_bodies) > 1 or len(arc_bodies) > 1:
        return BodyExtraction(reason="multiple_structured_bodies", candidate_count=len(json_bodies) + len(arc_bodies))
    if restricted:
        return BodyExtraction(reason="restricted_story_data")
    articles = []
    for node in soup.find_all("article"):
        content, count = _text_blocks(node, title)
        links = sum(len(a.get_text(" ")) for a in node.find_all("a"))
        if len(content) >= 40 and links / len(content) < .35:
            articles.append((content, count))
    unique_articles = list(dict.fromkeys(articles))
    if len(unique_articles) == 1:
        content, count = unique_articles[0]
        return BodyExtraction(content, "ARTICLE_BLOCKS", "body_found", count, 1)
    if len(unique_articles) > 1:
        return BodyExtraction(reason="multiple_article_candidates", candidate_count=len(unique_articles))
    # Do not flatten the entire page's div/span text into a pretend article.
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    content, count = _text_blocks(main if main is not None else soup, title, paragraphs_only=True)
    if content:
        return BodyExtraction(content, "MAIN_PARAGRAPHS" if main else "PAGE_PARAGRAPHS", "body_found", count, 1)
    return BodyExtraction(reason="empty_article_shell" if soup.find(id="article") is not None else "no_usable_body")
