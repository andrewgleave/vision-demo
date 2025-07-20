import logging
import asyncio
import base64
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ImageContent, function_tool
from livekit.plugins import google, noise_cancellation
from livekit.agents.voice import RunContext

from utils import load_prompt

logger = logging.getLogger("medical-vision-assistant")
logger.setLevel(logging.INFO)

load_dotenv()

# Global image handler tasks
_image_tasks = []
_current_session = None


def create_image_handler():
    """Create a global image handler that can be shared across agents"""

    def fn(reader, participant_identity):
        task = asyncio.create_task(_handle_image(reader, participant_identity))
        _image_tasks.append(task)
        task.add_done_callback(lambda t: _image_tasks.remove(t))

    return fn


async def _handle_image(reader, participant_identity):
    """Global image handler that forwards to current agent"""

    logger.info("Received image from %s: '%s'", participant_identity, reader.info.name)
    try:
        image_bytes = bytes()
        async for chunk in reader:
            image_bytes += chunk

        # Get the current session and agent
        global _current_session
        if _current_session and _current_session.current_agent:
            chat_ctx = _current_session.current_agent.chat_ctx.copy()
            chat_ctx.add_message(
                role="user",
                content=[
                    ImageContent(
                        image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    )
                ],
            )
            await _current_session.current_agent.update_chat_ctx(chat_ctx)
            logger.info("Medical image received and processed for analysis")
    except Exception as e:
        logger.error("Error processing medical image: %s", e)


@dataclass
class UserData:
    """Stores data and agents to be shared across the medical session"""

    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None

    def summarize(self) -> str:
        return "Medical office triage system with video analysis capabilities"


class BaseAgent(Agent):
    def __init__(self, instructions: str, **kwargs):
        super().__init__(
            instructions=instructions,
            **kwargs,
        )

    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata: UserData = self.session.userdata
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes(
                {"agent": agent_name}
            )

        chat_ctx = self.chat_ctx.copy()

        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        chat_ctx.add_message(
            role="system", content=f"You are the {agent_name}. {userdata.summarize()}"
        )

        await self.update_chat_ctx(chat_ctx)
        await self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance. Use the existing chat context to guide your response."
        )

    def _truncate_chat_ctx(
        self,
        items: list,
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list:
        """Truncate the chat context to keep the last n messages."""

        def _valid_item(item) -> bool:
            if (
                not keep_system_message
                and item.type == "message"
                and item.role == "system"
            ):
                return False
            if not keep_function_call and item.type in [
                "function_call",
                "function_call_output",
            ]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        while new_items and new_items[0].type in [
            "function_call",
            "function_call_output",
        ]:
            new_items.pop(0)

        return new_items

    async def _transfer_to_agent(self, name: str, context: RunContext) -> Agent:
        """Transfer to another agent while preserving context"""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]
        userdata.prev_agent = current_agent

        return next_agent


class TriageAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(instructions=load_prompt("triage_prompt.yaml"))

    @function_tool
    async def transfer_to_support(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("support", context)

    @function_tool
    async def transfer_to_billing(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("billing", context)


class SupportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(instructions=load_prompt("support_prompt.yaml"))

    @function_tool
    async def transfer_to_triage(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_billing(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("billing", context)


class BillingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(instructions=load_prompt("billing_prompt.yaml"))

    @function_tool
    async def transfer_to_triage(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_support(self, context: RunContext) -> Agent:
        await self.session.generate_reply()
        return await self._transfer_to_agent("support", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Register global image handler once
    ctx.room.register_byte_stream_handler("test", create_image_handler())

    userdata = UserData(ctx=ctx)
    triage_agent = TriageAgent()
    support_agent = SupportAgent()
    billing_agent = BillingAgent()

    # Register all agents in the userdata
    userdata.personas.update(
        {"triage": triage_agent, "support": support_agent, "billing": billing_agent}
    )

    session = AgentSession[UserData](
        userdata=userdata,
        llm=google.beta.realtime.RealtimeModel(
            # model="gemini-live-2.5-flash-preview",
            voice="Puck",
            temperature=0.8,
        ),
    )

    # Store session reference for global image handler
    global _current_session
    _current_session = session

    await session.start(
        agent=triage_agent,  # Start with the Medical Office Triage agent
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
