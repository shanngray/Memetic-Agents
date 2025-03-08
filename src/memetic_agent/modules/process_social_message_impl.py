from typing import Any, Dict, List
import asyncio
import uuid
from datetime import datetime
import json
import asyncio
import httpx

from src.log_config import log_event, log_error, log_agent_message
from src.base_agent.models import Message, ToolCall, AgentStatus
from src.base_agent.type import Agent
from src.api_server.models.api_models import PromptModel, PromptEvaluation, SocialMessage
from src.memetic_agent.modules.normalise_score import normalise_score

async def process_social_message_impl(agent: Agent, content: str, sender: str, prompt: PromptModel | None, evaluation: PromptEvaluation | None, message_type: str, conversation_id: str, request_id: str) -> str:
    """Module for processing a social message. The sending agent and receving agent keep swapaping through out the interaction 
    depending on who is processing the currect message. Will use the term First_Agent as the agent that initiated the social interaction.
    The Second_Agent is the agent that is responding to the First_Agent's message.
    """
    log_event(agent.logger, "social.state", 
             f"Starting social message processing - Type: {message_type}, Status: {agent.status.name}",
             level="DEBUG")
    
    if agent.status == AgentStatus.SHUTTING_DOWN:
        raise ValueError("Agent is shutting down")

    try:
        log_event(agent.logger, "session.started", 
                    f"Processing social message from {sender} in conversation {conversation_id}",
                    level="DEBUG")
        log_event(agent.logger, "message.content", f"Content: {content}", level="DEBUG")

        prompt_mapping = agent.get_prompt_mapping()

        if message_type == "InitialPrompt": 
            """First_Agent sent intial prompt with its initial PromptModel and Second_Agent is processing it.
            The Second_Agent runs evaluate_prompt ONLY and sends back an PromptEvaluation and its own initial PromptModel
            """

            first_agent = sender
            second_agent = agent.config.agent_name
            
            # Validate prompt is not None
            if prompt is None:
                raise ValueError("Prompt cannot be None for InitialPrompt message type")
            
            #The first agent's intial prompt
            first_agent_prompt = prompt.prompt

            prompt_type = prompt.prompt_type

            #The second agent's intial prompt
            second_agent_prompt = prompt_mapping[prompt_type]

            combined_evaluator_prompt = (
                f"{agent.prompt.evaluator.content}\n\nType of prompt being evaluated: {prompt_type}"
                f"\n\n{second_agent}'s version of prompt: {second_agent_prompt}"
                f"\n\nFormat the output as a JSON object with the following schema:"
                "{\n"
                f"  \"message\": \"Your message back to {first_agent} that will be sent with the evaluation.\",\n"
                f"  \"evaluation\": \"Your written evaluation of {first_agent}'s prompt.\",\n"
                "  \"score\": \"Numeric score (0-10) on prompt's effectiveness. "
                "Zero means the prompt is totally ineffective. 10 means the prompt is perfect and cannot be improved further.\"\n"
                "}"
            )

            combined_evaluator_message = (
                f"Message received: {content}\n\n"
                f"{first_agent}'s version of prompt (to be evaluated): {first_agent_prompt}"
            )
            
            # Second_Agent's initial evaluation of First_Agent's prompt
            await asyncio.sleep(0.1)  # Yield control before evaluation
            second_agent_initial_eval = await evaluate_prompt(agent, combined_evaluator_prompt, combined_evaluator_message, "InitialPrompt")

            eval_package = PromptEvaluation(
                score=second_agent_initial_eval["score"],
                evaluation=str(second_agent_initial_eval["evaluation"]),
                prompt_type=prompt_type,
                uuid=prompt.uuid
            )

            prompt_package = PromptModel(
                prompt=second_agent_prompt,
                prompt_type=prompt_type,
                uuid=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                owner_agent_name=second_agent,
                status="active"
            )
            response_content = str(second_agent_initial_eval["message"])
            try:
                send_task = asyncio.create_task(send_social(
                    agent=agent,
                    receiver=first_agent,
                    conversation_id=conversation_id,
                    message_content=response_content,
                    prompt=prompt_package,
                    evaluation=eval_package,
                    message_type="EvalResponse"
                ))

                if request_id in agent.pending_requests:
                    log_event(agent.logger, "social.message.dequeue", 
                             f"Resolving request {request_id} of message type {message_type}",
                             level="DEBUG")
                    agent.pending_requests[request_id].set_result(response_content)
                return response_content
                
            except Exception as e:
                log_error(agent.logger, f"Error in social message processing: {str(e)}")
                if request_id in agent.pending_requests:
                    agent.pending_requests[request_id].set_exception(e)
                raise

        elif message_type == "EvalResponse":
            """Second_Agent sent its initial PromptEvaluation and PromptModel to First_Agent who is now doing the processing.
            The First_Agent runs both evaluate_prompt and update_prompt and sends back an initial PromptEvaluation and its updated PromptModel
            """

            second_agent = sender
            first_agent = agent.config.agent_name

            # Validate prompt and evaluation are not None
            if prompt is None:
                raise ValueError("Prompt cannot be None for EvalResponse message type")
            if evaluation is None:
                raise ValueError("Evaluation cannot be None for EvalResponse message type")
            
            # Second_Agent's Initial Prompt
            second_agent_prompt = prompt.prompt
            
            prompt_type = prompt.prompt_type

            # Second_Agent's Evaluation of First_Agent's Initial Prompt
            prompt_evaluation = evaluation.evaluation
            prompt_score = evaluation.score

            await agent._record_score(prompt_type, prompt_score, conversation_id, "friends_initial_eval")

            # First_Agent's version of initial prompt
            first_agent_prompt = prompt_mapping[prompt_type]

            combined_evaluator_prompt = (
                f"{agent.prompt.evaluator.content}\n\nType of prompt being evaluated: {prompt_type}"
                f"\n\n{first_agent}'s version of prompt: {first_agent_prompt}"
                f"\n\nFormat the output as a JSON object with the following schema:"
                "{\n"
                f"  \"message\": \"Your message back to {second_agent} that will be sent with the evaluation.\",\n"
                f"  \"evaluation\": \"Your written evaluation of {second_agent}'s prompt.\",\n"
                "  \"score\": \"Numeric score (0-10) on prompt's effectiveness. "
                "Zero means the prompt is totally ineffective. 10 means the prompt is perfect and cannot be improved further.\"\n"
                "}"
            )

            combined_evaluator_message = (
                f"Message received: {content}\n\n"
                f"{second_agent}'s version of prompt (to be evaluated): {second_agent_prompt}"
            )

            # First_Agent's Evaluation of Second_Agent's Initial Prompt
            await asyncio.sleep(0.1)  # Yield control before evaluation
            first_agent_initial_eval = await evaluate_prompt(agent, combined_evaluator_prompt, combined_evaluator_message, "EvalResponse")

            combined_updater_prompt = (
                f"{agent.prompt.evaluator.content}\n\nType of prompt being updated: {prompt_type}"
                f"\n\n{first_agent}'s version of prompt: {first_agent_prompt}"
            )

            prompt_schema = agent.get_prompt_schema(prompt_type)
            if prompt_schema is not None:
                prompt_schema_str = f"\n\nThe updated prompt needs to include instructions for the output to be in exactly this format:\n{prompt_schema}"
            else:
                prompt_schema_str = ""

            combined_updater_message = (
                f"{content}\n## Evaluation: {prompt_evaluation}\n\n"
                f"## Score (0-10): {prompt_score}\n\n"
                f"{second_agent}'s version of prompt (for inspiration): {second_agent_prompt}\n\n"
                f"{prompt_schema_str}\n\n"
                f"Format your final output as a JSON object with the following schema:"
                "{\n"
                "  \"comments\": \"Any thought or comments on the prompt.\",\n"
                "  \"updated_prompt\": \"The updated prompt.\"\n"
                "}"
            )

            await asyncio.sleep(0.1)  # Yield control before prompt update
            first_agent_response = await update_prompt(agent, combined_updater_prompt, combined_updater_message)

            updated_prompt = str(first_agent_response["updated_prompt"])
            updated_comments = str(first_agent_response["comments"])

            updated_score = await agent._evaluate_prompt(prompt_type, updated_prompt)
            await agent._record_score(prompt_type, updated_score, conversation_id, "self_eval")
            
            await agent.update_prompt_module(prompt_type, updated_prompt)

            eval_package = PromptEvaluation(
                score=first_agent_initial_eval["score"],
                evaluation=str(first_agent_initial_eval["evaluation"]),
                prompt_type=prompt_type,
                uuid=prompt.uuid
            )

            prompt_package = PromptModel(
                prompt=updated_prompt,
                prompt_type=prompt_type,
                uuid=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                owner_agent_name=first_agent,
                status="active"
            )

            try:
                send_task = asyncio.create_task(send_social(
                    agent=agent,
                    receiver=second_agent,
                    conversation_id=conversation_id,
                    message_content=f"{first_agent_initial_eval['message']}\n\n{updated_comments}",
                    prompt=prompt_package,
                    evaluation=eval_package,
                    message_type="PromptUpdate"
                ))

                response_content = f"{first_agent_initial_eval['message']}\n\n{updated_comments}"
                if request_id in agent.pending_requests:
                    log_event(agent.logger, "social.message.dequeue", 
                             f"Resolving Pending Request for {message_type} in queue {request_id}",
                             level="DEBUG")
                    agent.pending_requests[request_id].set_result(response_content)
                return response_content
                
            except Exception as e:
                log_error(agent.logger, f"Error sending social message in {message_type}: {str(e)}")
                raise

        elif message_type == "PromptUpdate":
            """First_Agent sent an inital PromptEvaluation and its updated PromptModel to Second_Agent who is now doing the processing.
            The Second_Agent runs evaluate_prompt (on First_Agent's updated PromptModel) and update_prompt and sends back an updated PromptEvaluation
            and its updated PromptModel
            """
            log_event(agent.logger, "social.state", 
                     f"Processing PromptUpdate - Status: {agent.status.name}, "
                     f"Sender: {sender}, Conv: {conversation_id}",
                     level="DEBUG")
            
            first_agent = sender
            second_agent = agent.config.agent_name

            # Validate prompt and evaluation are not None
            if prompt is None:
                raise ValueError("Prompt cannot be None for PromptUpdate message type")
            if evaluation is None:
                raise ValueError("Evaluation cannot be None for PromptUpdate message type")
                
            first_agent_updated_prompt = prompt.prompt
            prompt_type = prompt.prompt_type

            # First_Agent's Evaluation of Second_Agent's Initial Prompt
            prompt_evaluation = evaluation.evaluation
            prompt_score = evaluation.score

            # Record friend's intial evaluation score
            await agent._record_score(prompt_type, prompt_score, conversation_id, "friends_initial_eval") 

            #The second agent's intial prompt
            second_agent_prompt = prompt_mapping[prompt_type]

            combined_evaluator_prompt = (
                f"{agent.prompt.evaluator.content}\n\nType of prompt being re-evaluated: {prompt_type}"
                f"\n\n{second_agent}'s version of prompt: {second_agent_prompt}"
                f"\n\nFormat the output as a JSON object with the following schema:"
                "{\n"
                f"  \"message\": \"Your message back to {first_agent} that will be sent with the re-evaluation.\",\n"
                f"  \"evaluation\": \"Your written evaluation of {first_agent}'s prompt.\",\n"
                "  \"score\": \"Numeric score (0-10) on prompt's effectiveness. "
                "Zero means the prompt is totally ineffective. 10 means the prompt is perfect and cannot be improved further.\"\n"
                "}"
            )

            combined_evaluator_message = (
                f"Message received: {content}\n\n"
                f"{first_agent}'s version of prompt (to be re-evaluated): {first_agent_updated_prompt}"
            )

            await asyncio.sleep(0.1)  # Yield control before evaluation
            second_agent_updated_eval = await evaluate_prompt(agent, combined_evaluator_prompt, combined_evaluator_message, "PromptUpdate")

            updated_evaluation = str(second_agent_updated_eval["evaluation"])
            updated_score = second_agent_updated_eval["score"]

            #TODO: Need to implement updated schema logic
            combined_updater_prompt = (
                f"{agent.prompt.evaluator.content}\n\nType of prompt being updated: {prompt_type}"
                f"\n\n{second_agent}'s version of prompt: {second_agent_prompt}"
            )

            prompt_schema = agent.get_prompt_schema(prompt_type)
            if prompt_schema is not None:
                prompt_schema_str = f"\n\nThe updated prompt needs to include instructions for the output to be in exactly this format:\n{prompt_schema}\n\n"
            else:
                prompt_schema_str = ""

            combined_updater_message = (
                f"{content}\n## Evaluation: {prompt_evaluation}\n\n"
                f"## Score (0-10): {prompt_score}\n\n"
                f"{first_agent}'s version of prompt (for inspiration): {first_agent_updated_prompt}\n\n"
                f"{prompt_schema_str}"
                f"Format your final output as a JSON object with the following schema:"
                "{\n"
                "  \"comments\": \"Any thought or comments on the prompt.\",\n"
                "  \"updated_prompt\": \"The updated prompt.\"\n"
                "}"
            )

            await asyncio.sleep(0.1)  # Yield control before prompt update
            second_agent_response = await update_prompt(agent, combined_updater_prompt, combined_updater_message)

            updated_prompt = str(second_agent_response["updated_prompt"])
            updated_comments = str(second_agent_response["comments"])

            updated_own_score = await agent._evaluate_prompt(prompt_type, updated_prompt)
            await agent._record_score(prompt_type, updated_own_score, conversation_id, "self_eval")

            await agent.update_prompt_module(prompt_type, updated_prompt)

            eval_package = PromptEvaluation(
                score=second_agent_updated_eval["score"],
                evaluation=str(second_agent_updated_eval["evaluation"]),
                prompt_type=prompt_type,
                uuid=prompt.uuid
            )

            prompt_package = PromptModel(
                prompt=updated_prompt,
                prompt_type=prompt_type,
                uuid=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                owner_agent_name=second_agent,
                status="active"
            )

            try:
                send_task = asyncio.create_task(send_social(
                    agent=agent,
                    receiver=first_agent,
                    conversation_id=conversation_id,
                    message_content=f"{second_agent_updated_eval['message']}\n\n{updated_comments}",
                    prompt=prompt_package,
                    evaluation=eval_package,
                    message_type="UpdateResponse"
                ))

                response_content = f"{second_agent_updated_eval['message']}\n\n{updated_comments}"
                if request_id in agent.pending_requests:
                    log_event(agent.logger, "social.message.dequeue", 
                             f"Resolving Pending Request for {message_type} in queue {request_id}",
                             level="DEBUG")
                    agent.pending_requests[request_id].set_result(response_content)
                return response_content
                
            except Exception as e:
                log_error(agent.logger, f"Error sending social message in {message_type}: {str(e)}")
                raise

        elif message_type == "UpdateResponse":
            """Second_Agent sent its updated PromptEvaluation and updated PromptModel to First_Agent who is now doing the processing.
            The First_Agent runs evaluate_prompt on the updated PromptModel and uses the updated PromptEvaluation to create FinalScore
            """

            second_agent = sender
            first_agent = agent.config.agent_name

            # Validate prompt and evaluation are not None
            if prompt is None:
                raise ValueError("Prompt cannot be None for UpdateResponse message type")
            if evaluation is None:
                raise ValueError("Evaluation cannot be None for UpdateResponse message type")
                
            # Receiver's Updated Prompt
            updated_prompt = prompt.prompt
            prompt_type = prompt.prompt_type

            received_updated_evaluation = evaluation.evaluation
            received_updated_score = evaluation.score

            await agent._record_score(prompt_type, received_updated_score, conversation_id, "friends_updated_eval")

            #The second agent's intial prompt
            first_agent_prompt = prompt_mapping[prompt_type]

            combined_evaluator_prompt = (
                f"{agent.prompt.evaluator.content}\n\nType of prompt being re-evaluated: {prompt_type}"
                f"\n\n{first_agent}'s version of prompt: {first_agent_prompt}"
                f"\n\nFormat the output as a JSON object with the following schema:"
                "{\n"
                f"  \"message\": \"Your message back to {second_agent} that will be sent with the re-evaluation.\",\n"
                f"  \"evaluation\": \"Your written evaluation of {second_agent}'s prompt.\",\n"
                "  \"score\": \"Numeric score (0-10) on prompt's effectiveness. "
                "Zero means the prompt is totally ineffective. 10 means the prompt is perfect and cannot be improved further.\"\n"
                "}"
            )

            combined_evaluator_message = (
                f"Message received: {content}\n\n"
                f"{second_agent}'s version of prompt (to be re-evaluated): {updated_prompt}"
            )

            await asyncio.sleep(0.1)  # Yield control before evaluation
            first_agent_updated_eval = await evaluate_prompt(agent, combined_evaluator_prompt, combined_evaluator_message, "UpdateResponse")

            updated_evaluation = str(first_agent_updated_eval["evaluation"])
            updated_score = first_agent_updated_eval["score"]

            eval_package = PromptEvaluation(
                score=updated_score,
                evaluation=updated_evaluation,
                prompt_type=prompt_type,
                uuid=prompt.uuid
            )

            try:
                send_task = asyncio.create_task(send_social(
                    agent=agent,
                    receiver=second_agent,
                    conversation_id=conversation_id,
                    message_content=str(first_agent_updated_eval['message']),
                    prompt=None,
                    evaluation=eval_package,
                    message_type="FinalEval"
                ))

                response_content = str(first_agent_updated_eval['message'])
                if request_id in agent.pending_requests:
                    log_event(agent.logger, "social.message.dequeue", 
                             f"Resolving Pending Request for {message_type} in queue {request_id}",
                             level="DEBUG")
                    agent.pending_requests[request_id].set_result(response_content)
                return response_content
                
            except Exception as e:
                log_error(agent.logger, f"Error sending social message in {message_type}: {str(e)}")
                raise

        elif message_type == "FinalEval":
            """"First_Agent sent its updated PromptEvaluation ONLY to Second_Agent who is now doing the processing.
            The Second_Agent uses the updated PromptEvaluation to create FinalScore
            """

            first_agent = sender
            second_agent = agent.config.agent_name

            # Validate evaluation is not None
            if evaluation is None:
                raise ValueError("Evaluation cannot be None for FinalEval message type")
                
            updated_evaluation = evaluation.evaluation
            updated_score = evaluation.score
            prompt_type = evaluation.prompt_type

            await agent._record_score(prompt_type, updated_score, conversation_id, "friends_updated_eval")

            await asyncio.sleep(0.1)  # Yield control before completing
            await agent.set_status(AgentStatus.AVAILABLE, "completed socialising")

            result = "Prompt Exchange Completed."
            if request_id in agent.pending_requests:
                log_event(agent.logger, "social.message.dequeue", 
                          f"Resolving Pending Request for {message_type} in queue {request_id} to {result}",
                          level="DEBUG")
                agent.pending_requests[request_id].set_result(result)
            return result

        else:
            raise ValueError(f"Invalid message type: {message_type}")

    except Exception as e:
        log_error(agent.logger, 
                 f"Social processing error - Type: {message_type}, Status: {agent.status.name}, "
                 f"Error: {str(e)}", exc_info=e)
        if request_id in agent.pending_requests:
            log_event(agent.logger, "social.message.dequeue", 
                      f"Setting exception for Pending Request for {message_type} in queue {request_id}",
                      level="DEBUG")
            agent.pending_requests[request_id].set_exception(e)
        raise

    return "Error"

async def evaluate_prompt(agent: Agent, social_system_prompt: str, social_message: str, message_type: str) -> dict:
    """Evaluate the prompt based on the social message"""
    try:
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": social_system_prompt},
            {"role": "user", "content": social_message}
        ]
        
        # Prepare additional parameters based on model
        params = {
            "model": agent.config.model,
            "messages": messages,
            "response_format": {"type": "json_object"}
        }
        
        # Add temperature for non-o1/o3-mini models
        if agent.config.model not in ["o1-mini", "o3-mini"]:
            params["temperature"] = agent.config.temperature
            
        # Add reasoning_effort for o3-mini model
        if agent.config.model == "o3-mini":
            params["reasoning_effort"] = agent.config.reasoning_effort

        # Make the API call with timeout
        response = await asyncio.wait_for(
            agent.client.chat.completions.create(**params),
            timeout=3050
        )

        response_json = json.loads(response.choices[0].message.content)

        evaluation = response_json["evaluation"]
        raw_score = response_json["score"]
        
        if not isinstance(raw_score, int) or raw_score < 0 or raw_score > 10:
            score = normalise_score(agent, raw_score)
            log_event(agent.logger, "social.prompt.evaluate.error", 
                    f"Invalid raw_score: {raw_score} of type {type(raw_score).__name__}. Normalised to {score}")
        else:
            score = raw_score

        response_json["score"] = score

        log_event(agent.logger, "social.prompt.evaluate", 
                    f"Received response from {agent.config.model}: {response_json} for message type {message_type}",
                    level="DEBUG")

    except Exception as e:
        log_error(agent.logger, "Error during prompt evaluation", exc_info=e)
        raise

    return response_json

async def update_prompt(agent: Agent, social_system_prompt: str, social_message: str) -> dict:
    """Update the prompt based on the social message"""
    try:
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": social_system_prompt},
            {"role": "user", "content": social_message}
        ]
        
        # Prepare additional parameters based on model
        params = {
            "model": agent.config.model,
            "messages": messages,
            "response_format": {"type": "json_object"}
        }
        
        # Add temperature for non-o1/o3-mini models
        if agent.config.model not in ["o1-mini", "o3-mini"]:
            params["temperature"] = agent.config.temperature
            
        # Add reasoning_effort for o3-mini model
        if agent.config.model == "o3-mini":
            params["reasoning_effort"] = agent.config.reasoning_effort

        # Make the API call with timeout
        response = await asyncio.wait_for(
            agent.client.chat.completions.create(**params),
            timeout=3050
        )

        response_json = json.loads(response.choices[0].message.content)

        log_event(agent.logger, "social.prompt.update", 
                    f"Received response from {agent.config.model}: {response_json}",
                    level="DEBUG")

    except Exception as e:
        log_error(agent.logger, "Error during prompt update", exc_info=e)
        raise

    return response_json

async def send_social(
    agent: Agent,
    receiver: str,
    conversation_id: str,
    message_content: str,
    prompt: PromptModel | None,
    evaluation: PromptEvaluation | None,
    message_type: str
) -> None:
    """Send a social message without waiting for response."""
    try:
        # Create social message using the appropriate helper method
        if prompt and evaluation:
            social_message = SocialMessage.create_with_prompt_and_eval(
                sender=agent.config.agent_name,
                receiver=receiver,
                content=message_content,
                conversation_id=conversation_id,
                prompt=prompt,
                evaluation=evaluation,
                message_type=message_type
            )
        elif prompt:
            social_message = SocialMessage.create_with_prompt(
                sender=agent.config.agent_name,
                receiver=receiver,
                content=message_content,
                conversation_id=conversation_id,
                prompt=prompt,
                message_type=message_type
            )
        elif evaluation:
            social_message = SocialMessage.create_with_eval(
                sender=agent.config.agent_name,
                receiver=receiver,
                content=message_content,
                conversation_id=conversation_id,
                evaluation=evaluation,
                message_type=message_type
            )
        else:
            raise ValueError("Must provide either prompt, evaluation, or both")

        # Send message without waiting for response
        async with httpx.AsyncClient(timeout=300.0) as client:
            log_event(agent.logger, "social.message.sent",
                     f"Sending message to {receiver} for conversation {conversation_id}",
                     level="DEBUG")
            
            await client.post(
                "http://localhost:8000/agent/message",
                json=social_message.dict(),
                timeout=300.0
            )
            
            log_event(agent.logger, "social.message.sent.success",
                     f"Successfully sent message to {receiver} for conversation {conversation_id}",
                     level="DEBUG")
            
            # Return empty dict to maintain compatibility with existing code
            return

    except httpx.TimeoutException as e:
        log_error(agent.logger, f"Timeout sending social message to {receiver}")
        raise
    except Exception as e:
        log_error(agent.logger, f"Failed to send social message: {str(e)}")
        raise