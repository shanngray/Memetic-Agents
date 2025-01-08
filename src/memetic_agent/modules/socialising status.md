# Socialising Status

## Incoming Messages
Agent has a status of "socialising" and receives a message from another agent.

If 'Social Message', it needs to be processed as a socialmessage.

What to do if an agent receives a non-standard message? ... for now send back a rote reply

## Outgoing Messages
When an agent is in "socialising" status, it will send out messages to other agents. Normally, the user/human initiates each
interaction when socialising the agent needs to initiate the interaction.

At start of socialising, we give each agent a random seed between 1-300 seconds. (won't work well for testing)

After a given random interval, the agent will send a message to another agent. (which agent is chosen is random)

The agent needs to lookup all other agents and make sure that it is sending a message to an agent that is in 'socialising' status.

If the agent is not in 'socialising' status, it will send back a rote message saying it can't accept the message at this time.

## Social Message Model

intro: string

Payload: json object
 - prompt:
 - prompt_type:
 - uuid:
 - timestamp:
 - owner_agent_name:
 - status: sender_initial / receiver_initial / sender_response / receiver_response

## Processing Social Messages

Receiving agent accepts message and responds with a message of its own. If the message is free format ...

If the message contains a structured element then the structure needs to be maintained as the agent modifies its code.



## Status Transitions

## Status History

## Status