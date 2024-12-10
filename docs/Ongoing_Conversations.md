<!-- AI Instructions -->
<!-- Instructions for AI are enclosed in <<>>. Please read and act accordingly. -->

To implement the new feature for ongoing conversations with/between agents, we need to address several key requirements. Here's a detailed plan outlining the steps required, the files that will need to be modified, and any additional implementation details.

## Plan for Implementing Ongoing Conversations

### 1. Enable Follow-Up Questions in Conversations

**Objective:** Allow users to ask follow-up questions and maintain context within a conversation.

**Steps:**
- **Modify `BaseAgent` Class:**
  - The `process_message` method already handles appending messages correctly
  - Add conversation loading from short-term memory at startup in `_load_memory`
  - Update error handling to gracefully handle cases where conversation history is in long-term memory

- **Update `live_interact.py`:**
  - Create a new conversation store class:
    ```python
    class ConversationStore:
        def __init__(self, file_path: Path):
            self.file_path = file_path
            self.conversations = {}
            self.load()
            
        def load(self) -> None:
            """Load conversations from JSON file."""
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.conversations = json.load(f)
                    
        def save(self) -> None:
            """Save conversations to JSON file."""
            with open(self.file_path, "w") as f:
                json.dump(self.conversations, f, indent=2)
                
        def add_conversation(self, conversation_id: str, name: str, participants: List[str]) -> None:
            self.conversations[conversation_id] = {
                "name": name,
                "participants": participants,
                "last_active": datetime.now().isoformat(),
                "created": datetime.now().isoformat()
            }
            self.save()
    ```

### 2. List and Re-Initiate Existing Conversations

**Objective:** Provide a way to list existing conversations and re-initiate an old one.

**Steps:**
- **Modify `BaseAgent` Class:**
  - Add method to load conversation from memory. If it returns False, the agent needs to search long-term memory:
    ```python
    async def load_conversation(self, conversation_id: str) -> bool:
        """Load conversation from memory, returns True if found."""
        try:
            # Try short-term memory first
            results = await self.memory.retrieve(
                query=f"conversation_id:{conversation_id}",
                collection_name="short_term",
                n_results=1,
                metadata_filter={"conversation_id": conversation_id}
            )
            if results:
                self.conversations[conversation_id] = self._parse_conversation(results[0]["content"])
                return True
                
            # Try long-term memory if not found
            results = await self.memory.retrieve(
                query=f"conversation_id:{conversation_id}",
                collection_name="long_term",
                n_results=1,
                metadata_filter={"conversation_id": conversation_id}
            )
            if results:
                self.conversations[conversation_id] = self._parse_conversation(results[0]["content"])
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            return False
    ```

- **Update `live_interact.py`:**
  - Add conversation management commands:
    - `/new` - Start new conversation
    - `/list` - List available conversations
    - `/resume <id>` - Resume existing conversation
    - `/exit_conv` - Exit current conversation
    - `/rename <name>` - Rename current conversation
  - <<live_interact asks for a conversation_id on startup. We need to stop this and instead give the user the option of starting a new conversation or loading an 
  existing one. When starting a new conversation we need to create a new conversation_id using a uuid. We also need to exit from a conversation back to main menu 
  and have this work separately from exiting the program.>>
  - Add conversation naming functionality:
    ```python
    async def generate_conversation_name(conversation_history: List[str]) -> str:
        """Use LLM to generate a witty one-liner name for the conversation."""
        prompt = f"""Based on this conversation, generate a witty one-liner name that captures its essence.
        The name should be memorable and unique, but not longer than 50 characters.
        
        Conversation:
        {conversation_history}
        
        Name:"""
        # Make LLM call and return generated name
    ```
  - Add conversation cleanup on exit: <<this needs to be persistent. We need to save the conversation details including the name, participants, last active, message count, and creation date to our json file. this should run when the user chooses to exit from a conversation>>
    ```python
    async def cleanup_conversation(conversation_id: str, history: List[str]) -> None:
        """Generate name and save conversation details before exit."""
        name = await generate_conversation_name(history)
        conversations[conversation_id]["name"] = name
        # Save updated conversations dictionary
    ```
### 3. Store and Retrieve Conversations from Memory
**Objective:** Ensure proper conversation storage and retrieval.

**Steps:**
- **Update `MemoryManager` Class:**
  - Add conversation-specific metadata support:
    ```python
    async def store_conversation(
        self,
        conversation_id: str,
        content: str,
        participants: List[str],
        collection_name: str = "short_term"
    ) -> str:
        """Store conversation with specific metadata."""
        metadata = {
            "conversation_id": conversation_id,
            "participants": ",".join(participants),
            "timestamp": datetime.now().isoformat(),
            "type": "conversation"
        }
        return await self.store(content, collection_name, metadata)
    ```

- **Update `BaseAgent` Class:**
  - Add conversation tracking:
    ```python
    self.active_conversations: Dict[str, Dict] = {
        "conversation_id": {
            "participants": set(),
            "last_active": datetime,
            "message_count": int,
            "name": str
        }
    }
    ```
  - Update `_save_memory` to use `store_conversation`
  - Add conversation cleanup on agent shutdown

### Implementation Notes

1. **Conversation Naming:**
   - Names are generated by LLM when a conversation is ended or paused
   - Names are stored in `live_interact.py`'s local conversation store
   - Each agent maintains its own list of conversations it participated in
   - Conversation uuids are used as keys to match conversation store and agents's list of active conversations

2. **Conversation Storage:**
   - Short-term memory: Recent and active conversations
   - Long-term memory: Archived conversations
   - JSON file: Conversation metadata and quick lookup
   - Memory dump files: Backup of all conversations

3. **Performance Considerations:**
   - Lazy loading of conversation history
   - Periodic cleanup of old conversations
   - Batch updates to JSON store

4. **Error Handling:**
   - Graceful degradation if memory store is unavailable
   - Recovery from corrupted conversation files
   - Automatic conversation backup on errors

### Files to be Modified

1. `src/base_agent/base_agent.py`:
   - Update memory loading/saving
   - Add conversation management methods
   - Improve error handling

2. `src/live_interact.py`:
   - Add conversation management commands
   - Implement ConversationStore class
   - Update command handling

3. `src/memory/memory_manager.py`:
   - Add conversation-specific storage methods
   - Improve metadata handling
   - Add batch operations support

This plan provides a comprehensive approach to implementing ongoing conversations while maintaining good practices around error handling, performance, and data persistence.
