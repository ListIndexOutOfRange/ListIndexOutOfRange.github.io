const output = document.getElementById("output");
const input = document.getElementById("input");

const SECRET_DATE = "09/05/2026";
let unlocked = false;

function appendHtmlLine(html = "") {
  const div = document.createElement("div");
  div.innerHTML = html;
  output.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
}

function appendTextLine(text = "") {
  appendHtmlLine(escapeHtml(text));
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function typeLines(lines, delayMs = 220) {
  for (const line of lines) {
    appendTextLine(line);
    await sleep(delayMs);
  }
}

function boot() {
  console.log("SECRET: <nickname>jet'aimedepuis<XX>ans");
  // Use HTML append for nicer emphasis later
  appendTextLine("Initializing secure terminal...");
  appendTextLine("Connection established.");
  appendTextLine("");
  appendTextLine("Enter the important date (dd/mm/yyyy):");
}

async function showInstructions() {
  appendTextLine("");
  appendHtmlLine('<span class="hl-ok">ACCESS GRANTED.</span>');
  appendTextLine("");

  await typeLines(
    [
      "Instructions:",
      "- Setup the video projector.",
      "- On your Mackbook, open Chrome.",
      "- In the upper bar, select 'Présentation', then 'Cast'.",
      "- Change 'Source' to 'Caster l'écran'.",
      "- Select XGIMI Horizon",
      "- Click on 'Tout l'écran', then 'Partager'.",
      "Now, open the 'Parks' box.",
    ],
    240
  );
}

input.addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;

  const value = input.value.trim();
  appendTextLine(`> ${value}`);
  input.value = "";

  if (unlocked) {
    appendTextLine("No further input required.");
    return;
  }

  if (value === SECRET_DATE) {
    unlocked = true;
    input.disabled = true; // optional: remove further interaction once unlocked
    await showInstructions();
  } else {
    appendTextLine("Incorrect date.");
    appendTextLine("Try again.");
  }
});

boot();
