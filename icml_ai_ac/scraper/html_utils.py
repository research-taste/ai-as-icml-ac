from __future__ import annotations

import html
import json
from html.parser import HTMLParser
from typing import Any


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    collapsed = " ".join(html.unescape(value).split())
    return collapsed or None


class Link:
    def __init__(self, href: str, text: str) -> None:
        self.href = href
        self.text = text


class HTMLMetadataParser(HTMLParser):
    """Small HTML extractor tailored to ICML virtual-site pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[Link] = []
        self.title: str | None = None
        self.json_ld: list[Any] = []
        self.abstract: str | None = None
        self._tag_stack: list[str] = []
        self._current_link_href: str | None = None
        self._current_link_parts: list[str] = []
        self._capture_title = False
        self._title_parts: list[str] = []
        self._capture_json_ld = False
        self._json_ld_parts: list[str] = []
        self._capture_abstract_depth = 0
        self._abstract_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): value for key, value in attrs}
        self._tag_stack.append(tag)

        if tag == "a" and attr_map.get("href"):
            self._current_link_href = attr_map["href"]
            self._current_link_parts = []
        elif tag == "title":
            self._capture_title = True
            self._title_parts = []
        elif tag == "script" and attr_map.get("type") == "application/ld+json":
            self._capture_json_ld = True
            self._json_ld_parts = []

        element_id = attr_map.get("id")
        classes = attr_map.get("class") or ""
        if (
            element_id in {"abstractText", "abstract", "paper-abstract"}
            or "abstract" in classes.split()
        ):
            self._capture_abstract_depth = 1
            self._abstract_parts = []
        elif self._capture_abstract_depth:
            self._capture_abstract_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_link_href is not None:
            text = clean_text(" ".join(self._current_link_parts)) or ""
            self.links.append(Link(self._current_link_href, text))
            self._current_link_href = None
            self._current_link_parts = []
        elif tag == "title" and self._capture_title:
            self.title = clean_text(" ".join(self._title_parts))
            self._capture_title = False
        elif tag == "script" and self._capture_json_ld:
            payload = "\n".join(self._json_ld_parts).strip()
            if payload:
                try:
                    self.json_ld.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
            self._capture_json_ld = False

        if self._capture_abstract_depth:
            self._capture_abstract_depth -= 1
            if self._capture_abstract_depth == 0:
                self.abstract = clean_text(" ".join(self._abstract_parts))

        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._current_link_href is not None:
            self._current_link_parts.append(data)
        if self._capture_title:
            self._title_parts.append(data)
        if self._capture_json_ld:
            self._json_ld_parts.append(data)
        if self._capture_abstract_depth:
            self._abstract_parts.append(data)


def parse_html_metadata(markup: str) -> HTMLMetadataParser:
    parser = HTMLMetadataParser()
    parser.feed(markup)
    return parser


def iter_json_ld_nodes(payloads: list[Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, dict):
            graph = payload.get("@graph")
            if isinstance(graph, list):
                nodes.extend(node for node in graph if isinstance(node, dict))
            nodes.append(payload)
        elif isinstance(payload, list):
            nodes.extend(node for node in payload if isinstance(node, dict))
    return nodes
