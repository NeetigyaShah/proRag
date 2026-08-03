// Soft "scan complete" tick for the scan-bed ingestion stage — pure Web
// Audio synthesis (a short filtered noise click + a quiet high blip), so no
// mp3 assets are needed. AudioContext is created lazily on first unmute
// (browsers block autoplay until a user gesture; the mute toggle is one).

let ctx: AudioContext | null = null;

function audioContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  try {
    if (!ctx) {
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      ctx = new Ctor();
    }
    if (ctx.state === "suspended") void ctx.resume();
    return ctx;
  } catch {
    return null; // Web Audio unavailable — sound is a garnish, never an error
  }
}

/** One scan tick: a soft noise click with a quiet 1.2kHz blip, played as the
 *  light bar finishes a page. ~90ms long, fire-and-forget. */
export function playScanTick(): void {
  const ac = audioContext();
  if (!ac) return;
  const now = ac.currentTime;

  // Click: band-passed noise, fast decay.
  const noise = ac.createBufferSource();
  const buffer = ac.createBuffer(1, ac.sampleRate * 0.045, ac.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  noise.buffer = buffer;
  const bandpass = ac.createBiquadFilter();
  bandpass.type = "bandpass";
  bandpass.frequency.value = 2200;
  bandpass.Q.value = 1.2;
  const noiseGain = ac.createGain();
  noiseGain.gain.setValueAtTime(0.16, now);
  noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.045);
  noise.connect(bandpass).connect(noiseGain).connect(ac.destination);
  noise.start(now);

  // Blip: quiet sine tick at the checkmark moment.
  const osc = ac.createOscillator();
  osc.type = "sine";
  osc.frequency.setValueAtTime(1250, now);
  osc.frequency.exponentialRampToValueAtTime(950, now + 0.07);
  const oscGain = ac.createGain();
  oscGain.gain.setValueAtTime(0.035, now);
  oscGain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
  osc.connect(oscGain).connect(ac.destination);
  osc.start(now);
  osc.stop(now + 0.09);
}
