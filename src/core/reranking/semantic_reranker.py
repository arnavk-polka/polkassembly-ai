import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SemanticReranker:

    def __init__(self, model_name="BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, chunks: list, top_k: int = 5):
        if not chunks:
            return chunks

        pairs = [(query, c["content"]) for c in chunks]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()
        
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        
        scored_chunks = [
            {**chunk, "semantic_score": float(score.item()) if hasattr(score, "item") else float(score)}
            for chunk, score in zip(chunks, scores)
        ]
        
        scored_chunks.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        return scored_chunks


def get_reranker():
    """Get the global reranker instance"""
    from ...app.query_pipeline import _get_reranker
    return _get_reranker()

