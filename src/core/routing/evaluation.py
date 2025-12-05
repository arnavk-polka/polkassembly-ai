import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .types import Route, RouterDecision

logger = logging.getLogger(__name__)


@dataclass
class RouterMetrics:
    total_queries: int = 0
    correct_routes: int = 0
    correct_networks: int = 0
    correct_proposal_indices: int = 0
    correct_needs: int = 0
    route_accuracy: float = 0.0
    network_accuracy: float = 0.0
    proposal_index_accuracy: float = 0.0
    needs_accuracy: float = 0.0
    overall_accuracy: float = 0.0


def evaluate_router_decision(
    predicted: RouterDecision,
    expected_route: str,
    expected_network: Optional[str] = None,
    expected_proposal_index: Optional[int] = None,
    expected_needs: Optional[List[str]] = None
) -> Dict[str, bool]:
    results = {
        "route_correct": False,
        "network_correct": False,
        "proposal_index_correct": False,
        "needs_correct": False
    }
    
    try:
        expected_route_enum = Route(expected_route.lower())
        results["route_correct"] = predicted.route == expected_route_enum
    except ValueError:
        results["route_correct"] = False
    
    if expected_network:
        results["network_correct"] = predicted.network == expected_network.lower()
    else:
        results["network_correct"] = predicted.network is None
    
    if expected_proposal_index is not None:
        results["proposal_index_correct"] = predicted.proposal_index == expected_proposal_index
    else:
        results["proposal_index_correct"] = predicted.proposal_index is None
    
    if expected_needs:
        expected_needs_set = set(n.lower().strip() for n in expected_needs)
        predicted_needs_set = set(n.lower().strip() for n in (predicted.needs or []))
        results["needs_correct"] = expected_needs_set == predicted_needs_set
    else:
        results["needs_correct"] = len(predicted.needs or []) == 0
    
    return results


def calculate_metrics(evaluation_results: List[Dict[str, bool]]) -> RouterMetrics:
    metrics = RouterMetrics()
    metrics.total_queries = len(evaluation_results)
    
    if metrics.total_queries == 0:
        return metrics
    
    for result in evaluation_results:
        if result.get("route_correct", False):
            metrics.correct_routes += 1
        if result.get("network_correct", False):
            metrics.correct_networks += 1
        if result.get("proposal_index_correct", False):
            metrics.correct_proposal_indices += 1
        if result.get("needs_correct", False):
            metrics.correct_needs += 1
    
    metrics.route_accuracy = metrics.correct_routes / metrics.total_queries
    metrics.network_accuracy = metrics.correct_networks / metrics.total_queries
    metrics.proposal_index_accuracy = metrics.correct_proposal_indices / metrics.total_queries
    metrics.needs_accuracy = metrics.correct_needs / metrics.total_queries
    
    metrics.overall_accuracy = (
        metrics.route_accuracy * 0.5 +
        metrics.network_accuracy * 0.2 +
        metrics.proposal_index_accuracy * 0.2 +
        metrics.needs_accuracy * 0.1
    )
    
    return metrics


def print_metrics(metrics: RouterMetrics):
    logger.info("=" * 60)
    logger.info("Router Evaluation Metrics")
    logger.info("=" * 60)
    logger.info(f"Total Queries: {metrics.total_queries}")
    logger.info(f"Route Accuracy: {metrics.route_accuracy:.2%} ({metrics.correct_routes}/{metrics.total_queries})")
    logger.info(f"Network Accuracy: {metrics.network_accuracy:.2%} ({metrics.correct_networks}/{metrics.total_queries})")
    logger.info(f"Proposal Index Accuracy: {metrics.proposal_index_accuracy:.2%} ({metrics.correct_proposal_indices}/{metrics.total_queries})")
    logger.info(f"Needs Accuracy: {metrics.needs_accuracy:.2%} ({metrics.correct_needs}/{metrics.total_queries})")
    logger.info(f"Overall Accuracy: {metrics.overall_accuracy:.2%}")
    logger.info("=" * 60)

