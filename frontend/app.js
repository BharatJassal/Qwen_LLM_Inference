const btn = document.getElementById("send");
const out = document.getElementById("output");
const status = document.getElementById("status");

// live-update the slider value labels
const temp = document.getElementById("temperature");
const topP = document.getElementById("topP");
temp.addEventListener("input", () => document.getElementById("tempValue").textContent = temp.value);
topP.addEventListener("input", () => document.getElementById("topPValue").textContent = topP.value);

btn.addEventListener("click", send);

async function send() {
  const prompt = document.getElementById("prompt").value;
  const maxTokens = parseInt(document.getElementById("maxTokens").value);

  btn.disabled = true;
  out.textContent = "";
  status.textContent = "Generating...";

  const start = performance.now();
  let tokenCount = 0;

  try {
    const resp = await fetch("http://localhost:8000/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: maxTokens,
        temperature: parseFloat(temp.value),
        top_p: parseFloat(topP.value),
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      out.textContent += decoder.decode(value);
      tokenCount++;
    }

    const elapsed = (performance.now() - start) / 1000;
    status.textContent = `Done · ${elapsed.toFixed(1)}s`;
  } catch (err) {
    status.textContent = "Error: " + err.message;
  } finally {
    btn.disabled = false;
  }
}