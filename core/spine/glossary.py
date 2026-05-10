"""Film-glossary spine impls."""

from __future__ import annotations

from typing import Optional


def get_film_term_definition(term: str) -> dict:
    """Look up a film/cinematography term in the glossary."""
    from core.film_glossary import get_term_definition

    if not term or not term.strip():
        return {
            "success": False,
            "error": "No term provided. Please specify a film term to look up.",
        }

    result = get_term_definition(term)

    if result:
        return {
            "success": True,
            "key": result["key"],
            "name": result["name"],
            "category": result["category"],
            "definition": result["definition"],
        }

    return {
        "success": False,
        "error": f"Term '{term}' not found in glossary.",
        "suggestion": "Try searching with search_glossary for partial matches.",
    }


def search_glossary(query: str, category: Optional[str] = None) -> dict:
    """Search the film glossary for a partial term match."""
    from core.film_glossary import (
        GLOSSARY_CATEGORIES,
        search_glossary as do_search,
    )

    if not query or not query.strip():
        return {"success": False, "error": "No search query provided."}

    if category and category != "All" and category not in GLOSSARY_CATEGORIES:
        return {
            "success": False,
            "error": f"Invalid category '{category}'.",
            "valid_categories": GLOSSARY_CATEGORIES,
        }

    results = do_search(query, category)

    if results:
        return {
            "success": True,
            "query": query,
            "category_filter": category,
            "result_count": len(results),
            "terms": [
                {
                    "key": r["key"],
                    "name": r["name"],
                    "category": r["category"],
                    "definition": r["definition"],
                }
                for r in results
            ],
        }

    return {
        "success": True,
        "query": query,
        "category_filter": category,
        "result_count": 0,
        "terms": [],
        "message": f"No terms found matching '{query}'.",
    }


__all__ = ["get_film_term_definition", "search_glossary"]
