"""自定义 LiveKit STT 插件：硅基流动收费的 Qwen3-Omni 多模态接口。

硅基流动的 /v1/audio/transcriptions 端点只支持两个免费共享池模型（排队、实例回收
会导致几十秒超时）。Qwen3-Omni 是收费多模态模型，必须走 /v1/chat/completions 的
audio_url 协议，因此无法直接用官方 openai STT 插件，这里自行实现。

本插件为非流式 STT：本地 VAD 检出一段话结束后，整段音频重采样为 16kHz 单声道
WAV，base64 内联提交给 chat/completions 转写。LiveKit 框架会自动用
stt.StreamAdapter 把它和 AgentSession 配置的 VAD 组合起来（见
voice/agent.py 的 default.stt_node）。
"""

from __future__ import annotations

import base64
import io
import wave

import httpx
from livekit import rtc
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from livekit.agents.stt import (
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STT,
    STTCapabilities,
)
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, merge_frames

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 让模型只做逐字转写，不要回答音频内容或加解释
_TRANSCRIBE_PROMPT = "请逐字转写音频中的语音内容，只输出转写文本，不要添加任何解释或标点以外的内容。"


class QwenOmniSTT(STT):
    """通过硅基流动 chat/completions 多模态接口调用 Qwen3-Omni 的非流式 STT。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = DEFAULT_MODEL,
        language: str = "zh",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )
        self._model = model
        self._language = language
        self._base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "siliconflow"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        frame = buffer if isinstance(buffer, rtc.AudioFrame) else merge_frames(buffer)
        wav_bytes = _encode_wav_16k_mono(frame)
        audio_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": _TRANSCRIBE_PROMPT},
                    ],
                }
            ],
            "stream": False,
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=conn_options.timeout,
            )
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"Qwen3-Omni STT request timed out: {e}") from e
        except httpx.HTTPError as e:
            raise APIConnectionError(f"Qwen3-Omni STT request failed: {e}") from e

        if not resp.is_success:
            # APIStatusError 会根据状态码自行判定是否可重试（429/5xx 可重试）
            raise APIStatusError(
                f"Qwen3-Omni STT returned {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                body=resp.text,
            )

        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        text = text.strip()
        if not text:
            # 空结果返回空 alternatives，StreamAdapter 会直接丢弃，不触发一轮对话
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT)

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(language=self._language, text=text),
            ],
        )


def _encode_wav_16k_mono(frame: rtc.AudioFrame) -> bytes:
    """把任意采样率/声道的 s16 帧重采样为 16kHz 单声道并封装成 WAV。"""
    if frame.sample_rate != 16000 or frame.num_channels != 1:
        resampler = rtc.AudioResampler(
            input_rate=frame.sample_rate,
            output_rate=16000,
            num_channels=1,
            quality=rtc.AudioResamplerQuality.QUICK,
        )
        out_frames = resampler.push(frame)
        out_frames.extend(resampler.flush())
        frame = merge_frames(out_frames)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(frame.data.tobytes() if hasattr(frame.data, "tobytes") else bytes(frame.data))
    return buf.getvalue()
