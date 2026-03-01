#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--input" && i + 1 < argv.length) {
      args.input = argv[i + 1];
      i += 1;
    } else if (token === "--output" && i + 1 < argv.length) {
      args.output = argv[i + 1];
      i += 1;
    } else if (token === "--threshold" && i + 1 < argv.length) {
      args.threshold = Number(argv[i + 1]);
      i += 1;
    } else if (token === "--screen-width" && i + 1 < argv.length) {
      args.screenWidth = Number(argv[i + 1]);
      i += 1;
    } else if (token === "--screen-height" && i + 1 < argv.length) {
      args.screenHeight = Number(argv[i + 1]);
      i += 1;
    }
  }
  return args;
}

function mean(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const valid = values
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x));
  if (valid.length === 0) {
    return null;
  }
  return valid.reduce((acc, x) => acc + x, 0) / valid.length;
}

function reasonToString(reason, recorderClass) {
  if (!reason || !recorderClass) {
    return "unknown";
  }
  if (reason === recorderClass.success) {
    return "success";
  }
  if (reason === recorderClass.fail) {
    return "fail";
  }
  if (reason === recorderClass.notEnoughProvidedData) {
    return "notEnoughProvidedData";
  }
  if (typeof reason === "symbol") {
    return reason.description || reason.toString();
  }
  return String(reason);
}

function sanitizeTrajectory(points) {
  if (!Array.isArray(points)) {
    return [];
  }
  const out = [];
  for (let i = 0; i < points.length; i += 1) {
    const row = points[i];
    if (!Array.isArray(row) || row.length < 3) {
      continue;
    }
    const x = Number(row[0]);
    const y = Number(row[1]);
    const t = Number(row[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(t)) {
      continue;
    }
    out.push([x, y, t]);
  }
  if (out.length < 4) {
    return [];
  }
  return out;
}

async function loadDelbot() {
  try {
    return require("@chrisgdt/delbot-mouse");
  } catch (requireErr) {
    try {
      const mod = await import("@chrisgdt/delbot-mouse");
      if (mod && mod.default) {
        return Object.assign({}, mod.default, mod);
      }
      return mod;
    } catch (importErr) {
      throw new Error(
        `Could not load @chrisgdt/delbot-mouse. require error: ${String(requireErr)}; import error: ${String(importErr)}`
      );
    }
  }
}

async function classifyOneTrajectory(delbot, model, points, threshold, screenWidth, screenHeight) {
  const clean = sanitizeTrajectory(points);
  if (clean.length < 4) {
    return {
      detected: false,
      is_human: false,
      p_bot: null,
      reason: "invalidTrajectory",
      n_records: 0,
    };
  }

  const recorder = new delbot.Recorder(screenWidth, screenHeight);
  let time0 = null;
  let lastT = -1;

  for (let i = 0; i < clean.length; i += 1) {
    const x = clean[i][0];
    const y = clean[i][1];
    const tRaw = clean[i][2];

    if (time0 === null) {
      time0 = tRaw;
    }

    let t = tRaw - time0;
    if (!Number.isFinite(t)) {
      t = 0;
    }
    if (t <= lastT) {
      t = lastT + 1;
    }
    lastT = t;

    recorder.addRecord({
      time: t,
      x,
      y,
      type: "Move",
    });
  }

  if (recorder.getRecords().length < 4) {
    return {
      detected: false,
      is_human: false,
      p_bot: null,
      reason: "notEnoughProvidedData",
      n_records: recorder.getRecords().length,
    };
  }

  let predictionList = [];
  try {
    predictionList = await recorder.getPrediction(model, false);
  } catch (_err) {
    predictionList = [];
  }

  const avgPBot = mean(predictionList);
  const result = await recorder.isHuman(model, threshold, false, false);
  const reason = reasonToString(result && result.reason, delbot.Recorder);
  const isHuman = Boolean(result && result.result);
  const pBot = avgPBot === null ? (isHuman ? 0.0 : 1.0) : avgPBot;
  const hasEnoughData = reason !== "notEnoughProvidedData";

  return {
    // DELBOT reason "fail" still returns a valid non-human classification.
    detected: hasEnoughData ? !isHuman : false,
    is_human: isHuman,
    p_bot: pBot,
    reason,
    n_records: recorder.getRecords().length,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) {
    throw new Error("Missing required --input <path> argument.");
  }

  const inputPath = path.resolve(args.input);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input file not found: ${inputPath}`);
  }

  const payload = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  const trajectories = Array.isArray(payload.trajectories) ? payload.trajectories : [];
  const threshold = Number.isFinite(args.threshold)
    ? args.threshold
    : Number.isFinite(payload.threshold)
      ? payload.threshold
      : 0.2;
  const screenWidth = Number.isFinite(args.screenWidth)
    ? args.screenWidth
    : Number.isFinite(payload.screen_width)
      ? payload.screen_width
      : 1920;
  const screenHeight = Number.isFinite(args.screenHeight)
    ? args.screenHeight
    : Number.isFinite(payload.screen_height)
      ? payload.screen_height
      : 1080;

  await import("@tensorflow/tfjs");
  const delbot = await loadDelbot();

  if (!delbot || !delbot.Recorder || !delbot.Models || !delbot.Models.rnn1) {
    throw new Error(
      "DELBOT exports are missing expected Recorder/Models.rnn1 API."
    );
  }

  const model = delbot.Models.rnn1;
  const results = [];
  for (let i = 0; i < trajectories.length; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    const entry = await classifyOneTrajectory(
      delbot,
      model,
      trajectories[i],
      threshold,
      screenWidth,
      screenHeight
    );
    results.push(entry);
  }

  const output = JSON.stringify({
    ok: true,
    threshold,
    screen_width: screenWidth,
    screen_height: screenHeight,
    n_trajectories: trajectories.length,
    results,
  });

  if (args.output) {
    fs.writeFileSync(path.resolve(args.output), `${output}\n`, "utf8");
  } else {
    process.stdout.write(`${output}\n`);
  }
}

main().catch((err) => {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`DELBOT helper failed: ${msg}\n`);
  process.exit(1);
});
