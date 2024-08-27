import logging
from celery import Celery
from datetime import datetime
from eduadvisor import model as edu_model
from eduadvisor import agents
import redis

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

celery_app = Celery("tasks", broker="redis://localhost:6379/0")
redis_client = redis.Redis(host="localhost", port=6379, db=0)


def send_process_message_to_user(
    state: edu_model.ConversationState,
    action: edu_model.StateAction,
    message_content: str,
):
    logger.debug(f"Sending process message to user: {message_content}")

    state = state.model_copy(
        update={
            "messages": state.messages
            + [
                edu_model.Message(
                    message_type=edu_model.MessageType.PROCESS,
                    state_action=action,
                    content=message_content,
                    timestamp=datetime.now(),
                )
            ]
        },
        deep=True,
    )

    channel_name = f"conversation:{state.conversation_id}"
    redis_client.publish(channel_name, state.model_dump_json())
    logger.debug(f"Message published to channel: {channel_name}")


@celery_app.task(name="process_action")
def process_action(
    conversation_state: edu_model.ConversationState | str,
    previous_action: edu_model.StateAction | None,
    action: edu_model.StateAction,
):
    logger.debug("Starting process_action task")

    if isinstance(conversation_state, str):
        state = edu_model.ConversationState.model_validate_json(conversation_state)
        logger.debug("Validated conversation state from JSON")
    else:
        state = conversation_state

    send_process_message_to_user(state, action, f"Processing your {action} request")
    logger.debug(f"Process message sent for action: {action}")

    state = agents.transition(
        state, previous_action, action, send_process_message_to_user
    )
    logger.debug("State transitioned")

    send_process_message_to_user(
        state, action, f"We finished the processing of your request"
    )
    logger.debug("Final process message sent")
