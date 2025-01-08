from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Optional

class ConversationStore:
    def __init__(self, file_path: Path):
        """Initialize the conversation store used by Live Interact CLI.
        This is a simple JSON file that stores conversations and their details.
        """
        self.file_path = file_path
        self.conversations: Dict[str, Dict] = {}
        self.load()
    
    def load(self) -> None:
        """Load conversations from JSON file."""
        if self.file_path.exists():
            with open(self.file_path) as f:
                self.conversations = json.load(f)
    
    def save(self) -> None:
        """Save conversations to JSON file."""
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self.conversations, f, indent=2)
    
    def add_conversation(self, conversation_id: str, name: str, participants: List[str]) -> None:
        """Add a new conversation to the store."""
        now = datetime.now().isoformat()
        self.conversations[conversation_id] = {
            "name": name,
            "participants": participants,
            "last_active": now,
            "created": now,
            "message_count": 0
        }
        self.save()
    
    def update_conversation(self, conversation_id: str, **kwargs) -> None:
        """Update conversation details."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].update(kwargs)
            self.conversations[conversation_id]["last_active"] = datetime.now().isoformat()
            self.save()
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation details by ID."""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self) -> List[Dict]:
        """List all conversations with their details."""
        return [
            {"id": conv_id, **details}
            for conv_id, details in self.conversations.items()
        ]
    
    def increment_message_count(self, conversation_id: str) -> None:
        """Increment the message count for a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["message_count"] += 1
            self.conversations[conversation_id]["last_active"] = datetime.now().isoformat()
            self.save() 