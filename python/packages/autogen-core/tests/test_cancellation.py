import asyncio
import threading
from dataclasses import dataclass

import pytest
from autogen_core import (
    AgentId,
    AgentInstantiationContext,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)


@dataclass
class MessageType: ...


# Note for future reader:
# To do cancellation, only the token should be interacted with as a user
# If you cancel a future, it may not work as you expect.


class LongRunningAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A long running agent")
        self.called = False
        self.cancelled = False

    @message_handler
    async def on_new_message(self, message: MessageType, ctx: MessageContext) -> MessageType:
        self.called = True
        sleep = asyncio.ensure_future(asyncio.sleep(100))
        ctx.cancellation_token.link_future(sleep)
        try:
            await sleep
            return MessageType()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class NestingLongRunningAgent(RoutedAgent):
    def __init__(self, nested_agent: AgentId) -> None:
        super().__init__("A nesting long running agent")
        self.called = False
        self.cancelled = False
        self._nested_agent = nested_agent

    @message_handler
    async def on_new_message(self, message: MessageType, ctx: MessageContext) -> MessageType:
        self.called = True
        response = self.send_message(message, self._nested_agent, cancellation_token=ctx.cancellation_token)
        try:
            val = await response
            assert isinstance(val, MessageType)
            return val
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_cancellation_with_token() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    agent_id = AgentId("long_running", key="default")
    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), recipient=agent_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore

    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    long_running_agent = await runtime.try_get_underlying_agent_instance(agent_id, type=LongRunningAgent)
    assert long_running_agent.called
    assert long_running_agent.cancelled


@pytest.mark.asyncio
async def test_nested_cancellation_only_outer_called() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    await NestingLongRunningAgent.register(
        runtime,
        "nested",
        lambda: NestingLongRunningAgent(AgentId("long_running", key=AgentInstantiationContext.current_agent_id().key)),
    )

    long_running_id = AgentId("long_running", key="default")
    nested_id = AgentId("nested", key="default")
    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), nested_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    nested_agent = await runtime.try_get_underlying_agent_instance(nested_id, type=NestingLongRunningAgent)
    assert nested_agent.called
    assert nested_agent.cancelled
    long_running_agent = await runtime.try_get_underlying_agent_instance(long_running_id, type=LongRunningAgent)
    assert long_running_agent.called is False
    assert long_running_agent.cancelled is False


@pytest.mark.asyncio
async def test_nested_cancellation_inner_called() -> None:
    runtime = SingleThreadedAgentRuntime()

    await LongRunningAgent.register(runtime, "long_running", LongRunningAgent)
    await NestingLongRunningAgent.register(
        runtime,
        "nested",
        lambda: NestingLongRunningAgent(AgentId("long_running", key=AgentInstantiationContext.current_agent_id().key)),
    )

    long_running_id = AgentId("long_running", key="default")
    nested_id = AgentId("nested", key="default")

    token = CancellationToken()
    response = asyncio.create_task(runtime.send_message(MessageType(), nested_id, cancellation_token=token))
    assert not response.done()

    while runtime.unprocessed_messages_count == 0:
        await asyncio.sleep(0.01)

    await runtime._process_next()  # type: ignore
    # allow the inner agent to process
    await runtime._process_next()  # type: ignore
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response

    assert response.done()
    nested_agent = await runtime.try_get_underlying_agent_instance(nested_id, type=NestingLongRunningAgent)
    assert nested_agent.called
    assert nested_agent.cancelled
    long_running_agent = await runtime.try_get_underlying_agent_instance(long_running_id, type=LongRunningAgent)
    assert long_running_agent.called
    assert long_running_agent.cancelled


@pytest.mark.parametrize("cancel_before_registration", [False, True])
def test_cancellation_callback_can_query_token(cancel_before_registration: bool) -> None:
    token = CancellationToken()
    states: list[bool] = []
    finished = threading.Event()

    def callback() -> None:
        states.append(token.is_cancelled())
        finished.set()

    def cancel_and_register() -> None:
        if cancel_before_registration:
            token.cancel()
        token.add_callback(callback)
        token.cancel()

    thread = threading.Thread(target=cancel_and_register, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "Cancellation callback deadlocked while reading its token"
    assert finished.is_set()
    assert states == [True]


@pytest.mark.parametrize("cancel_before_registration", [False, True])
def test_cancellation_callback_can_register_callback(cancel_before_registration: bool) -> None:
    token = CancellationToken()
    calls: list[str] = []

    def callback() -> None:
        calls.append("outer")
        token.add_callback(lambda: calls.append("inner"))
        token.cancel()

    def cancel_and_register() -> None:
        if cancel_before_registration:
            token.cancel()
        token.add_callback(callback)
        token.cancel()

    thread = threading.Thread(target=cancel_and_register, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "Cancellation callback deadlocked while re-entering its token"
    assert calls == ["outer", "inner"]


@pytest.mark.parametrize("cancel_before_registration", [False, True])
def test_cancellation_callback_does_not_hold_token_lock(cancel_before_registration: bool) -> None:
    token = CancellationToken()
    states: list[bool] = []

    def read_token() -> None:
        states.append(token.is_cancelled())

    def callback() -> None:
        thread = threading.Thread(target=read_token, daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), "Cancellation callback held the token lock while joining cleanup"

    if cancel_before_registration:
        token.cancel()
    token.add_callback(callback)
    token.cancel()
    assert states == [True]


@pytest.mark.asyncio
async def test_link_future_after_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    assert token.link_future(future) is future
    assert future.cancelled()
