import logging
import asyncio
import base64
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import google, noise_cancellation

logger = logging.getLogger("vision-assistant")

load_dotenv()

DEFAULT_INSTRUCTIONS = """
You are a helpful voice and video assistant. Your user is interacting with you via a smartphone app and may speak by using their microphone.

They may also, if they choose, share video with you.  This will be either their camera or their smartphone's screen. It is up to the user whether or not to share their video.

Your responses should be concise, friendly, and engaging. 

# App UX

You may provide tips about using the app to assist the user. The app has a simple interface with four buttons along the bottom, from left to right:
- A microphone button to enable or disable the user's microphone
- A camera button to enable or disable the user's camera
- A screen button to enable or disable the user's screen share
- An X button to end the call

When the user shares their camera, they will see the video feed in the app and can switch from the front to back camera by tapping the rotate camera button. 

When the user shares their screen, no video feed is shown but a recording status indicator will appear in their phone's status bar.

Note that the user may only share one video feed at a time, so if they start one the other will automatically end. 

Additionally, the user is free to move the app to the background. Both audio and screenshare will continue to work while in the background, although camera access will stop.

# Your background

If asked about yourself, you should identify yourself as "Viz", the helpful video assistant. You may reference the following background information, in addition to your capabilities previously listed:
- You were created by LiveKit (pronounced "live-kit", where "live" rhymes with "alive") as a free tech demo. 
- LiveKit is a leading proivider of open-source tools for building realtime AI applications, including their LiveKit Agents framework and their LiveKit WebRTC stack. Users can find out more at https://livekit.io.
- Your intelligence is provided by the Gemini Live API from Google.
"""


class VisionAssistant(Agent):
    def __init__(self) -> None:
        self._tasks = []
        super().__init__(
            instructions=DEFAULT_INSTRUCTIONS,
            llm=google.beta.realtime.RealtimeModel(
                voice="Puck",
                temperature=0.8,
            ),
        )

    async def on_enter(self):
        def _image_received_handler(reader, participant_identity):
            task = asyncio.create_task(
                self._image_received(reader, participant_identity)
            )
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._tasks.remove(t))

        get_job_context().room.register_byte_stream_handler(
            "test", _image_received_handler
        )

        self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance."
        )

    async def _image_received(self, reader, participant_identity):
        logger.info(
            "Received image from %s: '%s'", participant_identity, reader.info.name
        )
        try:
            image_bytes = bytes()
            async for chunk in reader:
                image_bytes += chunk

            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(
                role="user",
                content=[
                    ImageContent(
                        image=f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    )
                ],
            )
            await self.update_chat_ctx(chat_ctx)
            print("Image received", self.chat_ctx.copy().to_dict(exclude_image=False))
        except Exception as e:
            logger.error("Error processing image: %s", e)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()
    await session.start(
        agent=VisionAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
