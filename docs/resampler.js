// Rééchantillonneur en flux (sinc fenêtré de Kaiser, polyphase) : n'importe quel taux entier -> 16 kHz.
// Sortie identique quelle que soit la taille des blocs fournis ; -1 dB à 7 kHz, < -80 dB au-delà de 10 kHz.
export class StreamingResampler {
  constructor(inRate, outRate = 16000, { zeroCrossings = 16, cutoff = 0.94, beta = 8.6 } = {}) {
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const g = gcd(inRate, outRate);
    this.L = outRate / g; this.M = inRate / g;          // out[n] is at input position n*M/L
    const fc = cutoff * Math.min(1, outRate / inRate);   // low-pass cutoff (fraction of input Nyquist)
    const K = (this.K = Math.ceil(zeroCrossings / fc));  // taps per side
    const taps = (this.taps = 2 * K);
    const I0 = (x) => { let s = 1, t = 1; for (let k = 1; k < 50; k++) { t *= (x / (2 * k)) ** 2; s += t; } return s; };
    this.coef = new Float32Array(this.L * taps);         // one row per phase
    for (let p = 0; p < this.L; p++) {
      let sum = 0;
      for (let j = 0; j < taps; j++) {
        const d = p / this.L + K - 1 - j;                  // distance x - i, |d| <= K
        const sx = fc * d, s = sx === 0 ? 1 : Math.sin(Math.PI * sx) / (Math.PI * sx);
        const r = d / (K + 1), w = I0(beta * Math.sqrt(1 - r * r)) / I0(beta);
        sum += (this.coef[p * taps + j] = s * w);
      }
      for (let j = 0; j < taps; j++) this.coef[p * taps + j] /= sum; // unity DC gain
    }
    this.buf = new Float32Array(1 << 16); this.len = K;  // K zeros of left padding
    this.bufStart = -K; this.n = 0; this.totalIn = 0;
  }
  _append(x) {
    if (this.len + x.length > this.buf.length) {
      const nb = new Float32Array(Math.max(this.buf.length * 2, this.len + x.length));
      nb.set(this.buf.subarray(0, this.len)); this.buf = nb;
    }
    this.buf.set(x, this.len); this.len += x.length;
  }
  push(input) { this._append(input); this.totalIn += input.length; return this._run(Infinity); }
  flush() { this._append(new Float32Array(this.K + 1)); return this._run(Math.ceil(this.totalIn * this.L / this.M)); }
  _run(maxN) {
    const { L, M, K, taps, coef, buf } = this;
    const bufEnd = this.bufStart + this.len;
    // largest n with floor(n*M/L) + K < bufEnd
    let nEnd = Math.floor(((bufEnd - K - 1) * L + L - 1) / M) + 1;
    nEnd = Math.min(nEnd, maxN);
    const count = Math.max(0, nEnd - this.n), out = new Float32Array(count);
    for (let o = 0; o < count; o++, this.n++) {
      const num = this.n * M, ip = Math.floor(num / L);
      const row = (num - ip * L) * taps, base = ip - K + 1 - this.bufStart;
      let acc = 0;
      for (let j = 0; j < taps; j++) acc += coef[row + j] * buf[base + j];
      out[o] = acc;
    }
    const drop = Math.floor(this.n * M / L) - K + 1 - this.bufStart; // input no longer needed
    if (drop > 8192) { buf.copyWithin(0, drop, this.len); this.len -= drop; this.bufStart += drop; }
    return out;
  }
}
