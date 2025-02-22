# Memetic Agent Communication & Feedback Flow Diagram

1. **Message Dispatch**  
   The first agent prepares and sends a message via an API call and logs the dispatch while scheduling a background task for subsequent feedback evaluation.

2. **Message Reception**  
   The recipient agent receives the incoming message (via an HTTP POST endpoint), logs the details, and verifies the conversation context.

3. **Queue Insertion**  
   The received message is wrapped into a request with a unique ID, enqueued in the agent’s processing queue, and recorded with its queue position.

4. **Queue Processing**  
   An asynchronous task dequeues the request from the queue, logs statistical and debugging information, and initiates the message processing routine.

5. **Message Processing via LLM**  
   The agent processes the message by invoking the LLM (handling tool calls and iterations as needed), updating the conversation history, and generating a final response.

6. **Response Delivery**  
   The generated response is set on the corresponding future, signaling the completion of the processing, and the original sender receives the final answer.

7. **Feedback Evaluation & Transmission**  
   Concurrently, the first agent evaluates the quality of the response using an LLM evaluation method, constructs a feedback message, and sends this feedback to the respective agent via an API call.
