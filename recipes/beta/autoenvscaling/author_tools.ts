// The AutoEnvScaling agent's own tool: its task through the oracle and nop check. The gateway at REEF_AUTHOR_URL runs
// the check through Reef's generator service and holds every credential; this file holds none. A check outlasts one
// HTTP exchange (a build alone may take 20 minutes), so the tool starts it and polls it until it is done.
import { Type } from "typebox";

const POLL_MS = 20_000;

async function gateway(route, body, signal) {
  const response = await fetch(`${process.env.REEF_AUTHOR_URL}${route}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`${route} answered ${response.status}: ${text.slice(0, 2000)}`);
  return { content: [{ type: "text", text }], details: JSON.parse(text) };
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason ?? new Error("aborted"));
    }, { once: true });
  });
}

export default function (pi) {
  const deadline = Number(process.env.REEF_AUTHOR_DEADLINE_MS);
  const finishSeconds = Number(process.env.REEF_AUTHOR_FINISH_SECONDS);
  let latest = "No check yet.";

  pi.on("context", async (event) => {
    const remaining = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
    const status = [
      `Remaining time: ${remaining} seconds.`,
      remaining <= finishSeconds ? "Finish now: leave the best task you have in ./task and stop; no more checks." : "",
      `Latest task_check result: ${latest}`,
    ].filter(Boolean).join("\n");
    return { messages: [...event.messages, { role: "user", content: [{ type: "text", text: status }], timestamp: Date.now() }] };
  });

  // The author sets REEF_AUTHOR_REVIEWS=1 when a passing oracle check is followed by a review of the tests.
  const reviews = process.env.REEF_AUTHOR_REVIEWS === "1";
  pi.registerTool({
    name: "task_check",
    label: "Check the task",
    description:
      "Build ./task/environment and run the verifier twice: after the reference solution (oracle) and after doing " +
      "nothing (nop). The task passes when the oracle scores 1 and nop scores below 1" +
      (reviews
        ? ", and a reviewer then finds no requirement that tests/ checks and instruction.md does not state. " +
          "Answers with both rewards, the reviewer's findings and, "
        : ". Answers with both rewards and, ") +
      "when the task fails, the reason: the end of the build or trial error, or how the reference solution exited. " +
      "Takes minutes, a large build up to 20 minutes; the tool waits. No arguments.",
    promptSnippet: "Build the task and run its oracle and nop",
    promptGuidelines: ["Run task_check after writing the task and after every change, and fix what it reports."],
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, signal) {
      // /check starts a check, or names the one still running; /trial with "check <id>" polls it until it is done.
      let result = await gateway("/check", {}, signal);
      const checkId = result.details.check_id;
      while (result.details.state === "running") {
        if (Date.now() >= deadline) {
          const text = "The check is still running and the run's time is up: finish with the task as it stands.";
          return { content: [{ type: "text", text }], details: { state: "running", ran: false } };
        }
        await sleep(POLL_MS, signal);
        result = await gateway("/trial", { task: `check ${checkId}` }, signal);
      }
      const details = result.details;
      latest = JSON.stringify({
        check: details.check,
        ran: details.ran,
        passed: details.passed,
        stage: details.stage,
        oracle: details.oracle_reward,
        nop: details.nop_reward,
        checks_left: details.checks_left,
        review_findings:
          details.review === undefined ? undefined : details.review.ran ? details.review.findings.length : "not run",
      });
      return result;
    },
  });
}
