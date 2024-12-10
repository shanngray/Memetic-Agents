'''
ChromaDB CLI Tool
python -m src.utils.chroma_cli --list
python -m src.utils.chroma_cli --preview <collection_name>
python -m src.utils.chroma_cli --show <collection_name>
'''
import asyncio
import argparse
from typing import Optional
from pathlib import Path
import sys

from src.database.chroma_database import get_chroma_client

async def list_collections(client) -> None:
    """Display all collections with their statistics."""
    collections = client.list_collections()
    
    if not collections:
        print("No collections found.")
        return

    # Gather stats for each collection
    collection_stats = []
    for col in collections:
        count = col.count()
        peek = col.peek()
        collection_stats.append({
            "Name": col.name,
            "Documents": count,
            "Sample Fields": list(peek["metadatas"][0].keys()) if count > 0 and peek["metadatas"] else []
        })

    # Determine column widths
    widths = {
        "Name": max(len("Collection Name"), max(len(str(stats["Name"])) for stats in collection_stats)),
        "Documents": max(len("Document Count"), max(len(str(stats["Documents"])) for stats in collection_stats)),
        "Sample Fields": max(len("Metadata Fields"), max(len(", ".join(stats["Sample Fields"])) for stats in collection_stats))
    }

    # Create format string
    fmt = "| {:<{}} | {:^{}} | {:<{}} |"
    separator = "-" * (sum(widths.values()) + 7)  # +7 for the borders and spaces

    # Print header
    print("\nChromaDB Collections:")
    print(separator)
    print(fmt.format("Collection Name", widths["Name"],
                    "Document Count", widths["Documents"],
                    "Metadata Fields", widths["Sample Fields"]))
    print(separator)

    # Print data
    for stats in collection_stats:
        print(fmt.format(
            stats["Name"], widths["Name"],
            stats["Documents"], widths["Documents"],
            ", ".join(stats["Sample Fields"]) if stats["Sample Fields"] else "N/A", widths["Sample Fields"]
        ))
    print(separator + "\n")

async def preview_collection(client, collection_name: str) -> None:
    """Display preview (truncated) documents and metadata for a specific collection."""
    try:
        collection = client.get_collection(collection_name)
        results = collection.get()
        
        if not results["ids"]:
            print(f"\nNo documents found in collection: {collection_name}")
            return

        # Determine column widths
        id_width = max(len("ID"), max(len(str(id_)) for id_ in results["ids"]))
        content_width = max(len("Content"), max(len(str(doc)[:100]) for doc in results["documents"]))
        metadata_width = max(len("Metadata"), max(len(str(meta)) for meta in results["metadatas"]))

        # Create format string
        fmt = "| {:<{}} | {:<{}} | {:<{}} |"
        separator = "-" * (id_width + content_width + metadata_width + 7)

        # Print header
        print(f"\nPreviewing documents in collection: {collection_name}")
        print(separator)
        print(fmt.format("ID", id_width, "Content", content_width, "Metadata", metadata_width))
        print(separator)

        # Print data
        for id_, doc, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            content = (doc[:97] + "...") if len(doc) > 100 else doc
            print(fmt.format(
                str(id_), id_width,
                content, content_width,
                str(metadata), metadata_width
            ))
        print(separator + "\n")
            
    except Exception as e:
        print(f"Error accessing collection {collection_name}: {str(e)}")

async def show_collection(client, collection_name: str) -> None:
    """Display complete documents and metadata for a specific collection."""
    try:
        collection = client.get_collection(collection_name)
        results = collection.get()
        
        if not results["ids"]:
            print(f"\nNo documents found in collection: {collection_name}")
            return

        print(f"\nShowing all documents in collection: {collection_name}\n")
        for id_, doc, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            print(f"Document ID: {id_}")
            print("-" * 80)
            print("Content:")
            print(doc)
            print("\nMetadata:")
            print(metadata)
            print("=" * 80 + "\n")
            
    except Exception as e:
        print(f"Error accessing collection {collection_name}: {str(e)}")

async def confirm_action(action: str, target: str) -> bool:
    """Ask for user confirmation before destructive actions."""
    response = input(f"\nWARNING: Are you sure you want to {action} {target}? (y/n): ").lower()
    return response == 'y'

async def reset_collection(client, target: str) -> None:
    """Reset (clear) all documents from a collection or all collections."""
    try:
        if target == "all":
            collections = client.list_collections()
            if await confirm_action("reset", "ALL collections"):
                for collection in collections:
                    collection.delete(where={})
                print("\nSuccessfully reset all collections.")
        else:
            collection = client.get_collection(target)
            if await confirm_action("reset", f"collection '{target}'"):
                collection.delete(where={})
                print(f"\nSuccessfully reset collection: {target}")
                
    except Exception as e:
        print(f"Error resetting collection(s): {str(e)}")

async def delete_collection(client, target: str) -> None:
    """Delete a collection or all collections."""
    try:
        if target == "all":
            collections = client.list_collections()
            if await confirm_action("delete", "ALL collections"):
                for collection in collections:
                    client.delete_collection(collection.name)
                print("\nSuccessfully deleted all collections.")
        else:
            if await confirm_action("delete", f"collection '{target}'"):
                client.delete_collection(target)
                print(f"\nSuccessfully deleted collection: {target}")
                
    except Exception as e:
        print(f"Error deleting collection(s): {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="ChromaDB CLI Tool")
    parser.add_argument("--list", action="store_true", help="List all collections")
    parser.add_argument("--preview", type=str, help="Show preview of documents in specified collection")
    parser.add_argument("--show", type=str, help="Show complete documents in specified collection")
    parser.add_argument("--reset", type=str, help="Reset (clear) a collection or 'all' collections")
    parser.add_argument("--delete", type=str, help="Delete a collection or 'all' collections")
    
    args = parser.parse_args()
    
    client = await get_chroma_client()
    
    if args.list:
        await list_collections(client)
    elif args.preview:
        await preview_collection(client, args.preview)
    elif args.show:
        await show_collection(client, args.show)
    elif args.reset:
        await reset_collection(client, args.reset)
    elif args.delete:
        await delete_collection(client, args.delete)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
