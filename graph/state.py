from typing import List, Optional, TypedDict


class GraphState ( TypedDict ):
    question: str
    generation: str
    web_search: bool
    documents: Optional[List[str]]
