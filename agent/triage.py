# flake8: noqa: E501

import logging
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
from livekit.agents.llm import function_tool
from livekit.plugins import google, noise_cancellation
from livekit.agents.voice import RunContext

from utils import load_prompt

load_dotenv()

logger = logging.getLogger("medical-vision-assistant")
logger.setLevel(logging.INFO)

MODEL = "gemini-live-2.5-flash-preview"
TEMP = 0.8


@dataclass
class UserData:
    """Stores data and agents to be shared across the session"""

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

        chat_ctx = self.chat_ctx.copy(exclude_instructions=True)
        chat_ctx.add_message(
            role="system", content=f"You are the {agent_name}. {userdata.summarize()}"
        )

        await self.update_chat_ctx(chat_ctx)
        await self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance. Use the existing chat context to guide your response."
        )

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
        super().__init__(
            instructions=load_prompt("support_prompt.yaml"),
            llm=google.beta.realtime.RealtimeModel(
                model=MODEL,
                voice="Charon",  # custom voice for support agent
                temperature=TEMP,
            ),
        )

    @function_tool
    async def transfer_to_triage(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_billing(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("billing", context)


class BillingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=load_prompt("billing_prompt.yaml"),
            llm=google.beta.realtime.RealtimeModel(
                model=MODEL,
                voice="Kore",  # custom voice for billing agent
                temperature=TEMP,
            ),
        )

    @function_tool
    async def transfer_to_triage(self, context: RunContext) -> Agent:
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_support(self, context: RunContext) -> Agent:
        await self.session.generate_reply()
        return await self._transfer_to_agent("support", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData(ctx=ctx)
    triage_agent = TriageAgent()
    support_agent = SupportAgent()
    billing_agent = BillingAgent()

    # Register all agents in the userdata
    userdata.personas.update(
        {
            "triage": triage_agent,
            "support": support_agent,
            "billing": billing_agent,
        }
    )

    session = AgentSession[UserData](
        userdata=userdata,
        llm=google.beta.realtime.RealtimeModel(  # default model and voice
            model=MODEL,
            voice="Puck",
            temperature=TEMP,
        ),
    )

    await session.start(
        agent=triage_agent,  # Start with TriageAgent
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
