from celery import Celery
from datetime import datetime
from eduadvisor import model as edu_model
from eduadvisor import agents
import redis

celery_app = Celery("tasks", broker="redis://localhost:6379/0")
redis_client = redis.Redis(host="localhost", port=6379, db=0)


def send_process_message_to_user(
    state: edu_model.ConversationState,
    action: edu_model.StateAction,
    message_content: str,
):
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


@celery_app.task(name="process_action")
def process_action(
    conversation_state: edu_model.ConversationState | str,
    previous_action: edu_model.StateAction | None,
    action: edu_model.StateAction,
):

    if isinstance(conversation_state, str):
        state = edu_model.ConversationState.model_validate_json(conversation_state)
    else:
        state = conversation_state

    send_process_message_to_user(state, action, f"Processing your {action} request")

    state = agents.transition(
        state, previous_action, action, send_process_message_to_user
    )

    send_process_message_to_user(
        state, action, f"We finished the processing of your request"
    )
