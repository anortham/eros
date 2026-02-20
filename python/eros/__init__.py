"""
Eros — Semantic Code Intelligence MCP Server.

Named after the station in The Expanse where the protomolecule first
transformed raw data into understanding.

IMPORTANT: Keep this file minimal! Heavy imports here will block
the MCP handshake. All ML/embedding imports must be lazy-loaded
via lifecycle.py using asyncio.to_thread().
"""

__version__ = "0.1.0"
