import uuid
import os
import dotenv
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from chromadb import Collection

from src.log_config import log_event, log_error

dotenv.load_dotenv()

class MemoryManager:
    """Manages agent memory storage and retrieval using ChromaDB."""
    
    def __init__(self, agent_name: str, logger, chroma_client):
        self.agent_name = agent_name
        self.logger = logger
        self.memory_client = chroma_client
        self.collections: Dict[str, Collection] = {}

    async def initialize(self, collection_names: List[str] = ["short_term", "long_term"]) -> None:
        """Initialize the ChromaDB memory store with specified collections."""
        try:
            for name in collection_names:
                collection_name = f"{self.agent_name}_{name}"
                # Create collection if it doesn't exist
                self.collections[name] = self.memory_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": f"{name} memory store for {self.agent_name}"}
                )
                log_event(self.logger, "memory.initialized", 
                         f"Initialized collection: {collection_name} with {self.collections[name].count()} memories")
                         
            log_event(self.logger, "memory.initialized", 
                     f"Initialized {len(self.collections)} memory collections for {self.agent_name}")
                     
        except Exception as e:
            log_error(self.logger, f"Failed to initialize memory store: {str(e)}", exc_info=e)
            raise

    def _get_collection(self, collection_name: str) -> Collection:
        """Get a collection by name, raising error if not found."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection '{collection_name}' not found. Available collections: {list(self.collections.keys())}")
        return self.collections[collection_name]

    async def store(
        self, 
        content: str, 
        collection_name: str = "short_term",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a new memory in specified collection."""
        try:
            # Ensure collections are initialized
            if not self.collections:
                await self.initialize()
                
            # Ensure specific collection exists
            if collection_name not in self.collections:
                await self.initialize([collection_name])
                
            collection = self._get_collection(collection_name)
            memory_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "timestamp": datetime.now().isoformat(),
                "chroma_id": memory_id
            })
            
            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
            log_event(self.logger, "memory.stored", 
                     f"Stored memory {memory_id} in {collection_name}: {content[:100]}...", level="DEBUG")
            return memory_id
            
        except Exception as e:
            log_error(self.logger, f"Failed to store memory in {collection_name}: {str(e)}", exc_info=e)
            raise

    async def retrieve(
        self, 
        query: str,
        collection_names: Optional[Union[str, List[str]]] = None,
        n_results: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from specified collections."""
        try:
            # Ensure collections are initialized
            if not self.collections:
                await self.initialize()
                
            # Normalize collection_names to list and ensure they exist
            if collection_names is None:
                collection_names = list(self.collections.keys())
            elif isinstance(collection_names, str):
                collection_names = [collection_names]
                
            # Initialize any missing collections
            missing_collections = [name for name in collection_names if name not in self.collections]
            if missing_collections:
                await self.initialize(missing_collections)
                
            all_memories = []
            for name in collection_names:
                collection = self._get_collection(name)
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=metadata_filter
                )
                
                memories = [
                    {
                        "content": doc,
                        "metadata": {**metadata, "collection": name},
                        "relevance": 1 - distance
                    }
                    for doc, metadata, distance in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    )
                ]
                all_memories.extend(memories)
            
            # Sort combined results by relevance
            all_memories.sort(key=lambda x: x["relevance"], reverse=True)
            
            log_event(self.logger, "memory.retrieved", 
                     f"Retrieved {len(all_memories)} memories across {len(collection_names)} collections for query: {query}...",
                     level="DEBUG")
            return all_memories[:n_results]  # Return top N overall
            
        except Exception as e:
            log_error(self.logger, f"Failed to retrieve memories: {str(e)}", exc_info=e)
            raise

    async def delete(
        self, 
        memory_id: str,
        collection_name: Optional[str] = None
    ) -> None:
        """Delete a specific memory by ID from specified collection(s)."""
        if not self.collections:
            await self.initialize()
            
        try:
            if collection_name:
                collections = [self._get_collection(collection_name)]
            else:
                collections = list(self.collections.values())
                
            for collection in collections:
                collection.delete(ids=[memory_id])
                
            log_event(self.logger, "memory.deleted", 
                     f"Deleted memory {memory_id} from collection: {collection_name}")
            
        except Exception as e:
            log_error(self.logger, f"Failed to delete memory: {str(e)}", exc_info=e)
            raise

    async def store_categorized(
        self, 
        content: str,
        category: str,
        collection_name: str = "long_term",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a categorized memory in specified collection."""
        metadata = metadata or {}
        metadata["category"] = category
        return await self.store(content, collection_name, metadata)

    async def get_by_category(
        self,
        category: str,
        collection_names: Optional[Union[str, List[str]]] = None,
        query: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from specific category across specified collections."""
        metadata_filter = {"category": category}
        
        if query:
            return await self.retrieve(query, collection_names, n_results, metadata_filter)
        
        if not self.collections:
            await self.initialize()
            
        # Normalize collection_names to list
        if collection_names is None:
            collection_names = list(self.collections.keys())
        elif isinstance(collection_names, str):
            collection_names = [collection_names]
            
        all_memories = []
        for name in collection_names:
            collection = self._get_collection(name)
            results = collection.get(
                where=metadata_filter,
                limit=n_results
            )
            
            memories = [
                {
                    "content": doc,
                    "metadata": {**metadata, "collection": name},
                    "relevance": 1.0
                }
                for doc, metadata in zip(results["documents"], results["metadatas"])
            ]
            all_memories.extend(memories)
            
        return all_memories[:n_results]  # Return top N overall

    async def get_feedback_stats(
        self,
        sender: Optional[str] = None,
        timeframe: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get statistics about received feedback."""
        try:
            metadata_filter = {}
            if sender:
                metadata_filter["sender"] = sender
            if timeframe:
                cutoff = (datetime.now() - timeframe).isoformat()
                metadata_filter["timestamp"] = {"$gt": cutoff}

            collection = self._get_collection("feedback")
            results = collection.get(where=metadata_filter)
            
            scores = [metadata["score"] for metadata in results["metadatas"]]
            
            stats = {
                "count": len(scores),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            }
            
            return stats
            
        except Exception as e:
            log_error(self.logger, f"Failed to get feedback stats: {str(e)}")
            return {"error": str(e)}

    async def delete_by_metadata(
        self,
        collection_name: str,
        metadata_filter: Dict[str, Any]
    ) -> int:
        """Delete all memories matching the metadata filter from specified collection.
        
        Args:
            collection_name: Name of collection to delete from
            metadata_filter: Dictionary of metadata key-value pairs to match
            
        Returns:
            int: Number of memories deleted
        """
        try:
            collection = self._get_collection(collection_name)
            
            # Get matching records to count them
            matching_records = collection.get(where=metadata_filter)
            num_deleted = len(matching_records["ids"]) if matching_records["ids"] else 0
            
            # Delete matching records
            collection.delete(where=metadata_filter)
            
            log_event(self.logger, "memory.bulk_deleted", 
                     f"Deleted {num_deleted} memories from {collection_name} matching filter: {metadata_filter}")
            
            return num_deleted
            
        except Exception as e:
            log_error(self.logger, f"Failed to bulk delete memories: {str(e)}")
            raise