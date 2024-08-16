
---
title: Edu Bot
---
stateDiagram-v2
    BuildQuestion --> User: SeedQuestions
    User --> Coordinate: Question
    Coordinate --> Challenge: Question
    Coordinate --> User: SuggestedQuestions
    Challenge --> Coordinate: RefineQuestion
    Coordinate --> WebSearch: WebQuery
    WebSearch --> Coordinate: WebData
    Coordinate --> QueryLLM: Prompt
    QueryLLM --> User: Answer
---
title: Agent Flow
---
flowchart LR
    FRONT[HTMX Front End] -->|Question|API[API]
    FRONT[HTMX Front End] -->|Action|API
    API -->|Question| AGENT[Celery Agent Processor]
    API -->|Action| AGENT
    AGENT -->|State| AGENT
    AGENT -->|Action| AGENT
    AGENT -->|Process Message| API
    API -->|Process Message| FRONT

sequenceDiagram
    frontend->>/conversation: What questions do you have to me?
    /conversation->>frontend: Initial questions (conversation state) and conversation_id  
    frontend ->> /conversation/conversation_id: I chose this question and listen to feedback
    /conversation/conversation_id ->> frontend: Stream of next actions 
    frontend ->> /conversation/conversation_id/action: Chose next action
    /conversation/conversation_id ->> frontend: You found the answer
