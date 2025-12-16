import os
import asyncio
from typing import Any, Dict, List, Optional

from dkg import AsyncDKG  # type: ignore[import-untyped]
from dkg.providers import (  # type: ignore[import-untyped]
    AsyncBlockchainProvider,
    AsyncNodeHTTPProvider,
)


class DKGClientError(Exception):
    """Base exception for DKG client-related errors."""


class DKGConfigError(DKGClientError):
    """Raised when DKG configuration is missing or invalid."""


class DKGAssetNotFoundError(DKGClientError):
    """Raised when a requested Knowledge Asset cannot be found."""


class DKGQueryError(DKGClientError):
    """Raised when a SPARQL or graph query fails."""


_client: Optional[AsyncDKG] = None
_kagraph_cache: Dict[str, str] = {}


def _get_env_config() -> Dict[str, str]:
    """
    Read and validate required DKG configuration from environment variables.

    Returns
    -------
    dict
        Dictionary with "endpoint", "blockchain" and "environment" keys.

    Raises
    ------
    DKGConfigError
        If any required environment variable is missing.
    """
    endpoint = os.getenv("DKG_NODE_ENDPOINT")
    blockchain = os.getenv("DKG_BLOCKCHAIN")
    environment = os.getenv("DKG_ENVIRONMENT", "TESTNET")

    missing = []
    if not endpoint:
        missing.append("DKG_NODE_ENDPOINT")
    if not blockchain:
        missing.append("DKG_BLOCKCHAIN")

    if missing:
        raise DKGConfigError(
            f"Missing required DKG configuration env vars: {', '.join(missing)}"
        )

    return {"endpoint": endpoint, "blockchain": blockchain, "environment": environment}


def _get_client() -> AsyncDKG:
    """
    Lazily initialize and return a singleton AsyncDKG instance.

    Returns
    -------
    AsyncDKG
        Configured DKG client bound to the gateway node and blockchain.

    Raises
    ------
    DKGConfigError
        If required configuration is not available.
    """
    global _client
    if _client is None:
        config = _get_env_config()
        node_provider = AsyncNodeHTTPProvider(
            endpoint_uri=config["endpoint"],
            api_version="v1",
        )
        blockchain_provider = AsyncBlockchainProvider(
            config["blockchain"],
        )
        _client = AsyncDKG(
            node_provider,
            blockchain_provider,
            config={"max_number_of_retries": 300, "frequency": 2},
        )
    return _client


async def get_asset_by_ual(ual: str) -> Dict[str, Any]:
    """
    Fetch a Knowledge Asset by its UAL using the DKG Python SDK.

    Parameters
    ----------
    ual : str
        Knowledge Asset UAL, for example:
        ``did:dkg:otp:20430/0xcdb28e93ed340ec10a71bba00a31dbfcf1bd5d37/431747``.

    Returns
    -------
    dict
        Parsed JSON-LD representation of the Knowledge Asset assertion graph.

    Raises
    ------
    DKGAssetNotFoundError
        If the asset cannot be found for the provided UAL.
    DKGClientError
        If the SDK call fails for any other reason.
    """
    client = _get_client()
    try:
        response = await client.asset.get(ual)
    except Exception as exc:
        raise DKGClientError(f"Failed to fetch asset for UAL '{ual}': {exc}") from exc

    if not response:
        raise DKGAssetNotFoundError(f"No asset returned for UAL '{ual}'")

    if not isinstance(response, dict):
        raise DKGClientError(
            f"Unexpected response type for asset.get, expected dict, got {type(response)}"
        )

    if "assertion" in response:
        return response["assertion"]

    return response


async def resolve_ka_graph(ual: str) -> str:
    """
    Resolve the named graph URI (KaGraph) for a Knowledge Asset UAL.

    The resolution is performed via a SPARQL query against the DKG metadata
    describing which named graph contains the assertion triples for the
    provided Knowledge Asset.

    Results are cached in memory for the lifetime of the process.

    Parameters
    ----------
    ual : str
        Knowledge Asset UAL.

    Returns
    -------
    str
        Named graph IRI that contains the Knowledge Asset assertions.

    Raises
    ------
    DKGQueryError
        If the named graph cannot be resolved or the query fails.
    """
    if ual in _kagraph_cache:
        return _kagraph_cache[ual]

    client = _get_client()

    sparql_query = f"""
    PREFIX dkg: <https://ontology.origintrail.io/dkg/1.0#>
    SELECT ?kaGraph
    WHERE {{
        ?kc dkg:hasKnowledgeAsset <{ual}> ;
            dkg:hasNamedGraph ?kaGraph .
    }}
    LIMIT 1
    """

    try:
        response = await client.graph.query(sparql_query)
    except Exception as exc:
        raise DKGQueryError(
            f"Failed to resolve KaGraph for UAL '{ual}': {exc}"
        ) from exc

    if not isinstance(response, dict):
        raise DKGQueryError(
            f"Unexpected response type for graph.query, expected dict, got {type(response)}"
        )

    data = response.get("data", [])

    if not data:
        raise DKGQueryError(
            f"Could not resolve KaGraph for UAL '{ual}' (no data returned)"
        )

    first = data[0]
    ka_graph = first.get("kaGraph")
    if not isinstance(ka_graph, str) or not ka_graph:
        raise DKGQueryError(
            f"Resolved KaGraph value for UAL '{ual}' is empty or invalid"
        )

    _kagraph_cache[ual] = ka_graph
    return ka_graph


async def query_knowledge_asset(
    ual: str,
    sparql_where_clause: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Execute a SPARQL SELECT query against a single Knowledge Asset's assertion graph.

    Parameters
    ----------
    ual : str
        Knowledge Asset UAL to query.
    sparql_where_clause : str
        SPARQL WHERE clause body to be inserted inside the GRAPH block.
        Example::

            ?s <http://schema.org/name> ?o .
    limit : int, optional
        Maximum number of results to return, by default 50.

    Returns
    -------
    list of dict
        SPARQL bindings as returned by the DKG gateway triple store
        (SPARQL JSON ``results.bindings`` format).

    Raises
    ------
    DKGQueryError
        If the query execution fails.
    """
    ka_graph = await resolve_ka_graph(ual)
    client = _get_client()

    sparql_query = f"""
    SELECT ?s ?p ?o
    WHERE {{
        GRAPH <{ka_graph}> {{
            {sparql_where_clause}
        }}
    }}
    LIMIT {int(limit)}
    """

    try:
        response = await client.graph.query(sparql_query)
    except Exception as exc:
        raise DKGQueryError(
            f"Failed to execute SPARQL query for UAL '{ual}': {exc}"
        ) from exc

    if not isinstance(response, dict):
        raise DKGQueryError(
            f"Unexpected response type for graph.query, expected dict, got {type(response)}"
        )

    data = response.get("data", [])

    if not isinstance(data, list):
        raise DKGQueryError(
            f"Unexpected data structure in SPARQL response for UAL '{ual}'"
        )

    return data


async def _example_usage() -> None:
    """
    Minimal example of using this module.

    This function:
    1. Fetches a Knowledge Asset by UAL.
    2. Queries for ``http://schema.org/name`` values from that asset.
    """
    example_ual = (
        "did:dkg:otp:20430/"
        "0xcdb28e93ed340ec10a71bba00a31dbfcf1bd5d37/"
        "431747"
    )

    asset = await get_asset_by_ual(example_ual)

    where_clause = """
    ?s <http://schema.org/name> ?o .
    BIND(?s AS ?s)
    BIND(<http://schema.org/name> AS ?p)
    """

    results = await query_knowledge_asset(example_ual, where_clause, limit=50)

    _ = (asset, results)


if __name__ == "__main__":
    asyncio.run(_example_usage())


