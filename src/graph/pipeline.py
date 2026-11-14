from langgraph.graph import StateGraph, START, END
from src.graph.state import NewsLensState
from src.graph.nodes.ingest import ingest_node
from src.graph.nodes.classify import classify_node
from src.graph.nodes.micro_retrieval import micro_retrieval_node
from src.graph.nodes.credibility import credibility_node
from src.graph.nodes.claims import claim_extraction_node
from src.graph.nodes.verify import verify_node
from src.graph.nodes.synthesize import synthesize_and_persist_node


def should_verify_claims(state: NewsLensState) -> str:
    """Conditional edge: check if article contains check-worthy claims."""
    if state.get("has_checkworthy_claims", False):
        return "verify"
    return "synthesize"


def build_newslens_graph():
    """
    Constructs the NewsLens 3.0 LangGraph workflow:
    START
      -> Ingest
      -> Classify Macro
      -> Micro Retrieval & OOD
      -> Credibility & Linguistic Analysis
      -> Claim Extraction
      -> [Conditional]:
           - If check-worthy claims exist -> Verify (Live/Local Search + NLI + Aggregation) -> Synthesize
           - If no claims -> Synthesize directly
      -> END
    """
    builder = StateGraph(NewsLensState)

    # Add processing nodes
    builder.add_node("ingest", ingest_node)
    builder.add_node("classify", classify_node)
    builder.add_node("micro_retrieval", micro_retrieval_node)
    builder.add_node("credibility", credibility_node)
    builder.add_node("claims", claim_extraction_node)
    builder.add_node("verify", verify_node)
    builder.add_node("synthesize", synthesize_and_persist_node)

    # Define execution edges
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "classify")
    builder.add_edge("classify", "micro_retrieval")
    builder.add_edge("micro_retrieval", "credibility")
    builder.add_edge("credibility", "claims")

    # Conditional routing after claim extraction
    builder.add_conditional_edges(
        "claims",
        should_verify_claims,
        {
            "verify": "verify",
            "synthesize": "synthesize",
        },
    )

    builder.add_edge("verify", "synthesize")
    builder.add_edge("synthesize", END)

    # Compile executable graph
    return builder.compile()


# Singleton compiled graph
newslens_pipeline = build_newslens_graph()

