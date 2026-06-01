//! LiveKit *native* (libwebrtc) throughput/latency client — no Python GIL.
//! MODE=pub : encode+publish I420 frames at target FPS (libwebrtc encode).
//! MODE=sub : subscribe, count *decoded* frames from NativeVideoStream (libwebrtc decode).
//! Token generated externally (Python) and passed via LK_TOKEN.
use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use livekit::options::{TrackPublishOptions, VideoCodec, VideoEncoding};
use livekit::prelude::*;
use livekit::webrtc::prelude::*;
use livekit::webrtc::video_source::native::NativeVideoSource;
use livekit::webrtc::video_stream::native::NativeVideoStream;

fn ev(k: &str, d: &str) -> String {
    env::var(k).unwrap_or_else(|_| d.to_string())
}
fn now_us() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as i64
}

#[tokio::main]
async fn main() {
    let url = env::var("LK_URL").expect("LK_URL");
    let token = env::var("LK_TOKEN").expect("LK_TOKEN");
    let mode = ev("MODE", "sub");

    let (room, mut rx) = Room::connect(&url, &token, RoomOptions::default())
        .await
        .expect("connect failed");
    eprintln!("[rust-lk] connected mode={} room={}", mode, room.name());

    if mode == "pub" {
        let w: u32 = ev("W", "640").parse().unwrap();
        let h: u32 = ev("H", "480").parse().unwrap();
        let fps: f64 = ev("FPS", "200").parse().unwrap();
        let dur: f64 = ev("DURATION", "22").parse().unwrap();

        let source = NativeVideoSource::new(VideoResolution { width: w, height: h }, false);
        let track =
            LocalVideoTrack::create_video_track("cam", RtcVideoSource::Native(source.clone()));
        room.local_participant()
            .publish_track(
                LocalTrack::Video(track),
                TrackPublishOptions {
                    source: TrackSource::Camera,
                    video_codec: VideoCodec::H264,
                    video_encoding: Some(VideoEncoding {
                        max_framerate: ev("MAXFPS", "200").parse().unwrap(),
                        max_bitrate: 80_000_000,
                    }),
                    ..Default::default()
                },
            )
            .await
            .expect("publish failed");
        eprintln!("[rust-lk] publishing {}x{}@{} (H264) for {}s", w, h, fps, dur);

        let period = Duration::from_secs_f64(1.0 / fps);
        let t0 = Instant::now();
        let mut n: u64 = 0;
        let mut tick = tokio::time::interval(period);
        while t0.elapsed().as_secs_f64() < dur {
            tick.tick().await;
            let mut buf = I420Buffer::new(w, h);
            let (y, u, v) = buf.data_mut();
            let p = (n & 0xff) as u8; // varying content so the encoder actually works
            for b in y.iter_mut() {
                *b = p;
            }
            for b in u.iter_mut() {
                *b = 128;
            }
            for b in v.iter_mut() {
                *b = 128u8.wrapping_add(p);
            }
            let frame = VideoFrame {
                rotation: VideoRotation::VideoRotation0,
                timestamp_us: now_us(),
                frame_metadata: None,
                buffer: buf,
            };
            source.capture_frame(&frame);
            n += 1;
            if n % 200 == 0 {
                eprintln!("[rust-lk] pub pushed {} frames, {:.0} fps", n, n as f64 / t0.elapsed().as_secs_f64());
            }
        }
        eprintln!("[rust-lk] pub done, captured {} frames ({:.0} fps)", n, n as f64 / t0.elapsed().as_secs_f64());
    } else {
        let warmup: f64 = ev("WARMUP", "4").parse().unwrap();
        let dur: f64 = ev("DURATION", "12").parse().unwrap();
        eprintln!("[rust-lk] sub warmup={}s measure={}s", warmup, dur);

        let deadline = Instant::now() + Duration::from_secs_f64(warmup + dur + 30.0);
        while Instant::now() < deadline {
            let event = match tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
                Ok(Some(e)) => e,
                Ok(None) => break,
                Err(_) => continue,
            };
            if let RoomEvent::TrackSubscribed { track, .. } = event {
                if let RemoteTrack::Video(vt) = track {
                    let mut stream = NativeVideoStream::new(vt.rtc_track());
                    let t_start = Instant::now();
                    let mut measure_start: Option<Instant> = None;
                    let mut measured: u64 = 0;
                    while let Some(_frame) = stream.next().await {
                        if t_start.elapsed().as_secs_f64() < warmup {
                            continue;
                        }
                        if measure_start.is_none() {
                            measure_start = Some(Instant::now());
                        }
                        measured += 1;
                        if measure_start.unwrap().elapsed().as_secs_f64() >= dur {
                            break;
                        }
                    }
                    let secs = measure_start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(1.0);
                    let fps = measured as f64 / secs.max(1e-6);
                    println!("RESULT_JSON {{\"fps\": {:.1}, \"frames\": {}}}", fps, measured);
                    break;
                }
            }
        }
    }
    drop(room);
    std::process::exit(0); // avoid libwebrtc teardown segfault after measurement
}
