import asyncio, datetime, pathlib, sys, time
from backend.app.config import settings
from livekit import api, rtc
ROOM=f"gap-{sys.argv[1]}"; SR,CH,FRAME=48000,1,480
st={"quiet":0.0,"end":None,"gaps":[]}
async def main():
    room=rtc.Room()
    @room.on("track_subscribed")
    def on_t(t,p,par):
        if t.kind==rtc.TrackKind.KIND_AUDIO:
            async def d():
                async for _ in rtc.AudioStream(t):
                    now=time.time()
                    if st["end"] and now-st["quiet"]>0.5:
                        st["gaps"].append(round(now-st["end"],2)); st["end"]=None
                    st["quiet"]=now
            asyncio.create_task(d())
    tok=(api.AccessToken(settings.livekit_api_key,settings.livekit_api_secret)
         .with_identity("c").with_grants(api.VideoGrants(room_join=True,room=ROOM)).to_jwt())
    await asyncio.wait_for(room.connect(settings.livekit_url,tok),timeout=25)
    src=rtc.AudioSource(SR,CH)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("m",src),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(24)   # greeting
    for tag in ["1_english","2_name","3_service"]:
        pcm=pathlib.Path(f"/tmp/lk-speech/{tag}.pcm").read_bytes(); step=FRAME*2
        for i in range(0,len(pcm)-step,step):
            await src.capture_frame(rtc.AudioFrame(pcm[i:i+step],SR,CH,FRAME)); await asyncio.sleep(0.01)
        st["end"]=time.time(); st["quiet"]=time.time()
        for _ in range(800):
            await src.capture_frame(rtc.AudioFrame(b"\x00"*step,SR,CH,FRAME)); await asyncio.sleep(0.01)
            if st["end"] is None: break
        await asyncio.sleep(3)
    print(f"  {sys.argv[1]}: gap after you stop talking -> {st['gaps']}")
    await room.disconnect()
    lk=api.LiveKitAPI(url=settings.livekit_url.replace("wss://","https://"),
                      api_key=settings.livekit_api_key,api_secret=settings.livekit_api_secret)
    try: await lk.room.delete_room(api.DeleteRoomRequest(room=ROOM))
    except Exception: pass
    await lk.aclose()
asyncio.run(main())
