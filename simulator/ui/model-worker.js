// Runs the simulator model off the main thread, so the map and the controls stay
// responsive while a scenario is computed. Same-origin worker; the runtime itself
// comes from the CDN through importScripts, which workers may load cross-origin.
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js");
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

let session = null;

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      session ??= await ort.InferenceSession.create(m.url, { executionProviders: ["wasm"] });
      postMessage({ type: "ready" });
      return;
    }
    if (m.type === "run") {
      const { id, n, T, C, nn, nc, seq, num, cat } = m;
      const out = new Float32Array(n), B = 256, name = session.outputNames[0];
      for (let s = 0; s < n; s += B) {
        const b = Math.min(B, n - s);
        const r = await session.run({
          seq: new ort.Tensor("float32", seq.subarray(s * T * C, (s + b) * T * C), [b, T, C]),
          num: new ort.Tensor("float32", num.subarray(s * nn, (s + b) * nn), [b, nn]),
          cat: new ort.Tensor("int64", cat.subarray(s * nc, (s + b) * nc), [b, nc]),
        });
        out.set(r[name].data, s);
        postMessage({ type: "progress", id, done: s + b, n });
      }
      postMessage({ type: "done", id, out }, [out.buffer]);
    }
  } catch (err) {
    postMessage({ type: "error", id: m.id, message: String(err && err.message || err) });
  }
};
