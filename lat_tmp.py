import asyncio, os, pathlib, sys, time
from backend.app.config import settings
from livekit import api, rtc
TAG=sys.argv[1]; ROOM=f"lat-{TAG}"; SR,CH,FRAME=48000,1,480
state={"last_audio":0.0,"gaps":[]}
async def speak(src,tag):
    pcm=pathlib.Path(f"/tmp/lk-speech/{tag}.pcm").read_bytes(); step=FRAME*2
    for i in range(0,len(pcm)-step,step):
        await src.capture_frame(rtc.AudioFrame(pcm[i:i+step],SR,CH,FRAME)); await asyncio.sleep(0.01)
    done=time.time()
    for _ in range(60): await src.capture_frame(rtc.AudioFrame(b"\x00"*step,SR,CH,FRAME)); await asyncio.sleep(0.01)
    return done
async def main():
    lk=api.LiveKitAPI(url=settings.livekit_url.replace("wss://","https://"),
                      api_key=settings.livekit_api_key,api_secret=settings.livekit_api_secret)
    room=rtc.Room()
    @room.on("track_subscribed")
    def on_t(track,pub,p):
        if track.kind==rtc.TrackKind.KIND_AUDIO:
            async def drain():
                async for _ in rtc.AudioStream(track): state["last_audio"]=time.time()
            asyncio.create_task(drain())
    tok=(api.AccessToken(settings.livekit_api_key,settings.livekit_api_secret)
         .with_identity("c").with_grants(api.VideoGrants(room_join=True,room=ROOM)).to_jwt())
    await asyncio.wait_for(room.connect(settings.livekit_url,tok),timeout=25)
    src=rtc.AudioSource(SR,CH)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("m",src),
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(18)
    for t in ["1_english","2_name","3_service"]:
        state["last_audio"]=0.0
        done=await speak(src,t)
        for _ in range(120):
            await asyncio.sleep(0.05)
            if state["last_audio"]>done: break
        gap=state["last_audio"]-done
        if gap>0: state["gaps"].append(gap)
        await asyncio.sleep(6)
    g=state["gaps"]
    print(f"  {TAG}: response latency per turn -> {[f'{x:.2f}s' for x in g]}  median {sorted(g)[len(g)//2]:.2f}s" if g else f"  {TAG}: no measurement")
    await room.disconnect()
    try: await lk.room.delete_room(api.DeleteRoomRequest(room=ROOM))
    except Exception: pass
    await lk.aclose()
asyncio.run(main())
