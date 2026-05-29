# Security Policy

## Supported versions

LiOS is pre-1.0 and under active development. Security fixes are applied to the latest
state of the default branch. There is no long-term support branch yet.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on the [LiOS repository](https://github.com/cmriat/LiOS), or by contacting a maintainer
directly. Please include:

- a description of the issue and its impact,
- steps to reproduce (commands, configuration, environment),
- affected component (transport / signaling server / inference buffer) and version/commit.

We aim to acknowledge a report within a few business days and will coordinate a fix and
disclosure timeline with you.

## Scope

This project handles real-time media over WebRTC/SRTP and a Go signaling server. Relevant
areas include: signaling (SDP/ICE handling, room routing), media transport and encryption,
the CUDA-IPC inference buffer (shared-memory handles), and the control/state APIs under
`services/`. Issues in third-party dependencies (GStreamer, libwebrtc, PyTorch, coturn)
should be reported upstream, but feel free to flag them to us as well.
