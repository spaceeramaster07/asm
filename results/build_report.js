// Builds the project report (Word) from the results of the pipeline.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak,
} = require("docx");

const ROOT = "C:/Users/shamg/OneDrive/Desktop/ClaudeMaxxing/asm";
const PLOTS = path.join(ROOT, "results", "plots");
const OUT = path.join(ROOT, "results", "Barrier_Trading_Stochastic_Report.docx");

const FONT = "Times New Roman";
const TEXT_W = 9026; // A4 with 1" margins, in DXA

// ---------- inline markup: **bold**, *italic*
// subscripts X_t, r_f, S_{t-1} and superscripts e^(ga), rho^A -> real sub/superscript runs
const SUB = /([A-Za-z\u0370-\u03FF\u0394\]\)])(?:_(\{[^}]+\}|[A-Za-z0-9\u0370-\u03FF]+)|\^(\([^)]+\)|[A-Za-z0-9\u0370-\u03FF]+))/g;
function subRuns(s, opts) {
  const out = [];
  let last = 0, m;
  SUB.lastIndex = 0;
  while ((m = SUB.exec(s)) !== null) {
    out.push(new TextRun({ text: s.slice(last, m.index + 1), ...opts }));
    const raw = m[2] !== undefined ? m[2] : m[3];
    const inner = /^[{(]/.test(raw) ? raw.slice(1, -1) : raw;
    out.push(new TextRun({ text: inner, ...(m[2] !== undefined ? { subScript: true } : { superScript: true }), ...opts }));
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(new TextRun({ text: s.slice(last), ...opts }));
  return out.filter((r) => true);
}
function runs(text, base = {}) {
  text = text.replace(/(^|[^A-Za-z])f\*(?!\*)/g, "$1f∗");
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).filter((s) => s.length);
  return parts.flatMap((s) => {
    if (s.startsWith("**")) return subRuns(s.slice(2, -2), { bold: true, ...base });
    if (s.startsWith("*")) return subRuns(s.slice(1, -1), { italics: true, ...base });
    return subRuns(s, base);
  });
}
const P = (t, opt = {}) => new Paragraph({ children: runs(t), spacing: { after: 90 }, alignment: AlignmentType.JUSTIFIED, ...opt });
const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1 });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2 });
const B = (t) => new Paragraph({ children: runs(t), numbering: { reference: "bul", level: 0 }, spacing: { after: 40 } });
const N = (t) => new Paragraph({ children: runs(t), numbering: { reference: "num", level: 0 }, spacing: { after: 40 } });
const EQ = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 },
  children: subRuns(t, { font: "Cambria Math", size: 22 }),
});

// ---------- figures
let figNo = 0;
function FIG(file, caption, widthPx) {
  figNo += 1;
  const buf = fs.readFileSync(path.join(PLOTS, file + ".png"));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);   // PNG header
  const width = widthPx, height = Math.round(widthPx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: runs(`**Figure ${figNo}.** ${caption}`, { size: 19, color: "111111" }) }),
  ];
}

// ---------- tables
let tabNo = 0;
const HEAD_FILL = "34495E", STRIPE_FILL = "F3F5F8", RULE = "D5DAE1";
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(caption, header, rows, widths, aligns) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const al = (i) => (aligns ? aligns[i] : (i === 0 ? "left" : "center")) === "left" ? AlignmentType.LEFT : AlignmentType.CENTER;
  const cell = (t, w, i, kind) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: kind === "head" ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" }
           : kind === "stripe" ? { fill: STRIPE_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 120, right: 120 },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: al(i), spacing: { after: 0 },
      children: runs(t, kind === "head" ? { size: 19, bold: true, color: "FFFFFF" } : { size: 19 }) })],
  });
  return [
    new Paragraph({ spacing: { before: 160, after: 60 }, keepNext: true, alignment: AlignmentType.CENTER,
      children: runs(`**Table ${tabNo}.** ${caption}`, { size: 19, color: "111111" }) }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      borders: { top: { style: BorderStyle.SINGLE, size: 8, color: HEAD_FILL },
                 bottom: { style: BorderStyle.SINGLE, size: 8, color: HEAD_FILL },
                 left: NONE, right: NONE, insideVertical: NONE,
                 insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: RULE } },
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], i, "head")) }),
             ...rows.map((r, k) => new TableRow({ cantSplit: true,
               children: r.map((c, i) => cell(c, widths[i], i, k % 2 === 1 ? "stripe" : "plain")) }))],
    }),
    new Paragraph({ spacing: { after: 140 }, children: [] }),
  ];
}

// =============================================================== content
const C = [];

// ---------- Title + abstract
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: [new TextRun({ text: "Stochastic Models of a Take-Profit / Stop-Loss Trade and a Kelly Betting Game",
    bold: true, size: 32, font: FONT })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Theory, Real-Data Validation and Bet Sizing on the Nifty 50 and S&P 500",
    italics: true, size: 23, font: FONT })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 40 },
  children: [new TextRun({ text: "Sambhav Manohar          Venkat Rahul          Sham Gagan Siddharth", size: 24, font: FONT })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Applied Stochastic Models (ASM) Project, 2026     |     Domain: Stock Market", size: 21, font: FONT })] }));
C.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 1 } }, spacing: { after: 120 }, children: [] }));

C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "Abstract", bold: true, size: 24, font: FONT })] }));
C.push(P("We study a simple trading rule: buy a stock index and sell it once it has gained 20% or lost 10%. As a stochastic process this is the gambler's-ruin problem, a price wandering between two barriers until it hits one. The take-profit and stop-loss levels fix the odds and payoffs of each trade, a gain of 20% against a loss of about 10%, so repeated trading becomes a repeated gamble, and the remaining question is how much of one's wealth to stake on each round, which the Kelly criterion answers. We model daily prices with a biased random walk, geometric Brownian motion and Merton's jump-diffusion, derive the chance of winning, the expected length of a trade and the growth-optimal bet, and test these predictions on the Nifty 50 (2007–2025) and the S&P 500 (1962–2025). Each model is refitted monthly on the previous five years only, so every comparison is a genuine forecast.", { indent: { left: 567, right: 567 } }));
C.push(P("Simulations reproduce the theory once daily price checks are modelled. On real data, the jump-diffusion describes the size of daily moves, and the extra loss when a stop is jumped over, far better than the other models, but no model captures the alternation of calm and turbulent periods. Simulated traders then use each model to size bets over a hundred trades, in model-generated markets, in markets rebuilt from real returns and on the actual history. Betting the full optimal amount is a poor choice, because it depends on an average return that five years of data cannot pin down. Betting half of it does better everywhere, the jump-diffusion gives the best bet sizes, and holding the index grows wealth fastest, with deep temporary losses.", { indent: { left: 567, right: 567 } }));

// ---------- 1 Introduction
C.push(H1("1. Introduction"));
C.push(P("Stop-loss and take-profit orders are among the most common tools in trading. A trader who commits to selling an index at a fixed gain or loss faces two questions: how often will the gain come first, and how much should be bet on each trade? Probability theory answers both. For a random walk the first is the gambler's-ruin problem, for Brownian motion a first-passage problem with a closed-form solution, and the second is answered by the Kelly criterion, the bet that makes wealth grow fastest in the long run."));
C.push(P("This project tests whether those answers hold on real markets, following the full pipeline from the trading rule to data, model, mathematical analysis, simulation, validation and interpretation. The working hypothesis is that three standard models of log prices are good enough to predict the trade and to size bets on it. We test it on the Nifty 50, India's main stock index, with the longer S&P 500 history as an independent check."));

// ---------- 2 Problem statement
C.push(H1("2. Problem Statement"));
C.push(P("Let S_t be the index's closing price on day t and X_t = ln(S_t / S_0) its log return since the trade was opened. The trade closes at the first daily close at which X has risen to b = ln 1.2 (a 20% gain) or fallen to a = ln 0.9 (a 10% loss), or after two years (504 trading days) if neither has happened:"));
C.push(EQ("τ = min{ t : X_t ≥ b  or  X_t ≤ a },     b = ln 1.2,  a = ln 0.9."));
C.push(P("Because prices are only checked once a day, the price can fall well past −10% between closes, so a stopped trade can lose more than 10%. The project sets out to:"));
C.push(B("derive, for each model, the probability that the gain is reached first, the expected duration of a trade and the distribution of its final return;"));
C.push(B("check that the simulations reproduce those results;"));
C.push(B("test how well each model describes real daily returns and real trades;"));
C.push(B("work out how much of one's wealth each model says to bet, and test betting strategies over a long series of trades."));

// ---------- 3 Literature
C.push(H1("3. Literature Review"));
C.push(P("The exit problem for a random walk between two barriers is the gambler's ruin, treated in Feller [1]; Karlin and Taylor [2] give the continuous-time version for Brownian motion using martingale arguments. Geometric Brownian motion became the standard model of asset prices with Black and Scholes [3], and Merton [4] extended it with sudden jumps to explain large price moves. Broadie, Glasserman and Kou [5] showed how to correct continuous-time barrier formulas when the barrier is only checked at discrete times, which is exactly our situation."));
C.push(P("Real returns have heavier tails than the normal distribution, and their volatility comes in clusters [6]; the Ljung–Box test [7] is a standard way to detect such dependence. Forecasts of a whole probability distribution are judged with the probability integral transform [8], probability forecasts with proper scoring rules such as the log score and the Brier score [9, 10], and confidence intervals for dependent data come from block bootstraps [11, 12]. For betting, Kelly [13] introduced the growth-optimal bet, Thorp [14] applied it to games and markets, and MacLean, Thorp and Ziemba [15] explain why a fraction of the Kelly bet is safer when inputs are uncertain."));

// ---------- 4 Data
C.push(H1("4. Data Description"));
C.push(P("We use daily closing prices from Yahoo Finance, stored in fixed files so that every result can be reproduced. Daily log returns r_t = ln(S_t / S_{t−1}) are the input to all models."));
C.push(...TABLE("The two data sets.", ["", "Nifty 50 (main study)", "S&P 500 (validation)"], [
  ["Period", "Sept 2007 – Dec 2025", "Jan 1962 – Dec 2025"],
  ["Trading days", "4,487", "16,108"],
  ["Period on which everything is tested", "2017 – 2025", "1994 – 2025"],
  ["Interest rate on cash (assumed)", "6.0% per year", "2.5% per year"],
], [3300, 2863, 2863]));
C.push(P("The early years of each series only supply the first five years of history for fitting; they are never used to judge the models. All checks use closing prices, which are reliable for both indices."));

// ---------- 5 Mathematical background
C.push(H1("5. Mathematical Background"));
C.push(H2("5.1 Random walk"));
C.push(P("In the simplest model the log price moves up or down by a fixed step each day, up with probability p and down with probability q = 1 − p. If the barriers are A steps below and B steps above the starting point, conditioning on the first step gives the gambler's-ruin formula, and Wald's identity gives the expected duration:"));
C.push(EQ("P(win) = (1 − ρ^A) / (1 − ρ^(A+B)),   ρ = q/p,      E[τ] = (B·P(win) − A·(1 − P(win))) / (p − q)."));
C.push(H2("5.2 Geometric Brownian motion"));
C.push(P("Under geometric Brownian motion, dS = μS dt + σS dW, the log price is a Brownian motion with drift ν = μ − σ²/2. Because exp(γX_t) with γ = −2ν/σ² is a martingale, stopping it at the exit time gives the probability of reaching the upper barrier first, and Wald's identity gives the expected exit time:"));
C.push(EQ("P(win) = (1 − e^(γa)) / (e^(γb) − e^(γa)),      E[τ] = (b·P(win) + a·(1 − P(win))) / ν."));
C.push(P("This gives a useful benchmark: if the price has no tendency to rise (μ = 0), the trader wins exactly one time in three, because the gain is twice as far away as the loss. The formula assumes continuous watching; Broadie, Glasserman and Kou [5] showed that for daily checks it becomes accurate again if both barriers are pushed outward by about 0.58 σ√Δt."));
C.push(H2("5.3 Merton jump-diffusion"));
C.push(P("Merton's model adds jumps to the Brownian motion. Jumps arrive at an average rate λ per year and move the log price by a normally distributed amount, so a day's return is a mixture of normal distributions:"));
C.push(EQ("f(y) = Σ_n  e^(−λΔt) (λΔt)^n / n!  ·  φ( y ;  ν_{d}Δt + nμ_J ,  σ_d²Δt + nσ_J² )."));
C.push(P("This gives fatter tails than a single normal distribution, and because a jump can carry the price past a barrier, the exit problem has no simple formula."));
C.push(H2("5.4 An exact solution for daily checks"));
C.push(P("The formulas above either assume continuous watching or apply only to the random walk, and the jump-diffusion has no formula at all. Our trade, however, is checked once a day. To get exact daily-check answers for all three models, we solve the exit problem as a Markov chain. The interval between the barriers is divided into 1,200 cells, and the model's daily distribution gives the probability of moving between cells or past either barrier. With Q the matrix of moves between cells and u the probabilities of exiting at the upper barrier, the win probabilities p and expected durations T solve"));
C.push(EQ("(I − Q) p = u ,      (I − Q) T = 1 ."));
C.push(P("Stepping the chain forward day by day gives the full distribution of a trade's length and return, including how far beyond the stop a loss is filled. This method serves three purposes in the project: it is the exact benchmark against which the simulations are checked (Section 9), it supplies each model's forecasts of a real trade's outcome, length and loss (Section 10), and it gives the outcome distribution from which the Kelly bet is computed (Section 11)."));
C.push(H2("5.5 The Kelly criterion"));
C.push(P("A trader puts a fraction f of wealth into each trade and keeps the rest in cash. If R is the trade's return and T its length, wealth is multiplied by M(f) ≈ 1 + fR, plus interest on cash (or minus interest on borrowing when f > 1), less trading costs. Since wealth compounds, what matters is the expected logarithm of this factor per year, and the Kelly fraction maximises it:"));
C.push(EQ("f* = argmax_f  E[ ln M(f) ] / E[ T + 1 ] ."));
C.push(P("The growth curve rises, peaks at f* and falls. Betting half the Kelly amount keeps about three quarters of the extra growth, and over-betting costs much more than under-betting. If a large enough loss is possible, a high enough bet can wipe out the account in one trade; we call that level the *ruin cliff*."));

// ---------- 6 Stochastic model
C.push(H1("6. Stochastic Model"));
C.push(P("All three models are fitted to reproduce the average and spread of daily log returns exactly, so they differ only in the shape of the daily distribution: two values for the random walk, a normal distribution for Brownian motion, and a normal plus jumps for the jump-diffusion."));
C.push(B("**Random walk:** the step size and the probability of an up-step are chosen to match the average and variance of daily returns."));
C.push(B("**Geometric Brownian motion:** the drift and volatility are the average and standard deviation of daily returns, scaled to a year."));
C.push(B("**Jump-diffusion:** days with unusually large moves (more than three standard deviations) are treated as jumps; their frequency and size give the jump parameters, and the remaining variation gives the Brownian part."));
C.push(P("Because markets change, every model is refitted at the start of each month on the previous five years and used only to predict what follows. Five years was chosen over three and seven on the early data only, so the choice could not be tuned to the test period."));

// ---------- 7 Methodology
C.push(H1("7. Methodology"));
C.push(P("The work proceeds in four stages."));
C.push(P("**Theory against simulation.** For each model we simulate half a million trades and compare win rates and durations with the exact results of Section 5, checking the mathematics and the code before any data is used."));
C.push(P("**Models against daily returns.** Each day, the month's model gives a probability distribution for that day's return before it happens. We record how likely the model found the actual return, and transform the return into the standard normal value it should have if the model were right. Over thousands of days, these values show whether a model gets the size of moves right and whether it misses any pattern over time."));
C.push(P("**Models against real trades.** For every start day, each model forecasts the chance of winning, losing or timing out, the trade's length and the loss on a stop. We compare these forecasts with what happened and with a no-model alternative, the share of recent trades that won. Because neighbouring start days share most of their future, uncertainty is measured with block bootstraps."));
C.push(P("**Betting over many trades.** Finally, simulated traders use the models to size their bets over careers of 100 consecutive trades. The rules of this game and the strategies compared are described in Section 11."));
C.push(P("Each rule is played 2,000 times on the same price paths in three kinds of market: markets generated by each model with fixed true parameters; a market rebuilt from randomly chosen three-month blocks of real history (a stationary bootstrap), which keeps real crashes and calm-and-storm patterns; and the actual history."));

// ---------- 8 Implementation
C.push(H1("8. Computational Implementation"));
C.push(P("The project is written in Python with NumPy, SciPy, pandas and Matplotlib, as a small package with modules for the models, the exact formulas and Markov-chain solver, the trade simulator, the real-data scanner, the statistical tests, the Kelly calculations, the bootstrap and the betting game. A short sequence of scripts runs the pipeline, and fixed random seeds make every result reproducible."));
C.push(P("Two steps keep the computation manageable. The exact forecasts for every month and model are computed once, in parallel, and stored. And because the traders need a Kelly fraction before each of 200,000 trades per market, these are computed in advance: for every combination of drift and volatility on a grid (and every jump rate for the jump-diffusion), the Markov-chain method gives the exact distribution of a trade's length and return, the growth per year is evaluated for each bet including interest, borrowing and trading costs, and the best bet is stored. During the game the bet is read off by interpolating between neighbouring entries. Automated tests check the formulas, the solver, the bootstrap and the cost accounting, and confirm that no decision uses future information."));

// ---------- 9 Simulation results
C.push(H1("9. Simulation Results"));
C.push(H2("9.1 The simulations reproduce the theory"));
C.push(P("We simulated half a million trades with each model. The random walk matched the gambler's-ruin formula, and Brownian motion and the jump-diffusion matched the exact daily-check solution, all within random error; the Broadie–Glasserman–Kou correction matched the win rate and left only a fraction of a day of error in the average length. The textbook continuous-time formula was the exception: with daily checks, simulated trades win about 1.5 percentage points more often and last about 15 days longer than it predicts. For every exact comparison the error shrinks as one over the square root of the number of trades (Figure 1)."));
C.push(...FIG("q1_convergence_nifty", "Error of the simulated estimates as the number of simulated trades grows. The exact comparisons fall steadily; the textbook continuous formula (dashed) levels off at a fixed bias.", 560));
C.push(P("The textbook formula fails because it assumes the price is watched continuously, whereas a daily check lets the price cross a barrier during the day unseen, so trades last longer and win slightly more often. The Broadie–Glasserman–Kou correction fixes this by moving each barrier outward by β σ √Δt with β ≈ 0.5826; for σ = 16% this is about 0.0059 in log terms, turning the barriers into roughly +20.7% and −10.5%."));

// ---------- 10 Real data results
C.push(H1("10. Real Data Results"));
C.push(H2("10.1 Daily returns: the size of moves"));
C.push(P("The jump-diffusion describes real daily returns much better than Brownian motion on both indices, mostly because of crises such as 2008 and 2020, when Brownian motion treats a fall of several percent in a day as practically impossible and the jump-diffusion treats it as rare but expected. In Figure 2, returns transformed under Brownian motion bend away from the line in both tails, meaning extreme days happen far more often than it allows, while under the jump-diffusion they follow the line. The random walk cannot be scored this way, as it allows only two daily returns."));
C.push(...FIG("q2_qq_nifty", "Real daily returns transformed by each model, against the normal distribution they should follow if the model were right (Nifty). Brownian motion (orange) misses the extreme days; the jump-diffusion (green) captures them.", 310));

C.push(H2("10.2 How long a trade lasts"));
C.push(P("Each forecast also gives the expected length of a trade, which we compare with how long real trades took (Table 2)."));
C.push(...TABLE("Average length of a trade in trading days: real trades (with 95% confidence interval) and each model's forecast.", ["", "Nifty 50", "S&P 500"], [
  ["Real trades", "224 (153 – 302)", "215 (177 – 257)"],
  ["Random walk", "173", "168"],
  ["Brownian motion", "185", "179"],
  ["Jump-diffusion", "**191**", "**183**"],
], [3626, 2700, 2700]));
C.push(P("All three models expect trades to end one to two months sooner than they did, the jump-diffusion coming closest and the random walk furthest away. Every forecast lies within the uncertainty of the real average on the Nifty 50, and only the random walk falls outside it on the S&P 500. The average length is therefore judged reasonably well; the difficulty is the spread around it, examined below."));

C.push(H2("10.3 Forecasting the trade"));
C.push(P("All three models forecast the trade's outcome better than the no-model alternative on the S&P 500 (Figure 3); on the Nifty 50, nine years are too few to separate them from chance. The three models give almost identical forecasts, because the chance of reaching +20% before −10% depends mainly on the average return and volatility, which all three share."));
C.push(...FIG("q3_scores_spx", "How much better each model forecasts the trade's outcome than the no-model alternative on the S&P 500, with 95% confidence intervals. Positive log score and negative Brier score mean better forecasts.", 560));
C.push(P("The models share one error in timing (Figure 4): real trades end very quickly or very slowly more often than predicted. The cause is volatility clustering. The models treat each day as independent of the last, but in real markets turbulent days follow turbulent days, a dependence the Ljung–Box test [7] confirms decisively for both indices. In turbulent periods a barrier is hit within days; in calm ones the price can drift for a year."));
C.push(...FIG("q3_pit_spx", "How the actual length of each trade compares with each model's forecast (S&P 500). Correct forecasts would give a flat histogram at 10%; the raised ends show too many very short and very long trades.", 560));
C.push(P("The models differ clearly in the size of losses (Table 3). Because stops are checked only at the close, real stopped trades lose noticeably more than 10%, sometimes much more. The random walk never loses more than its step and Brownian motion rarely gaps far; only the jump-diffusion reproduces both the typical extra loss and the occasional large one."));
C.push(...TABLE("Average loss on stopped trades, and the worst 1% of them.", ["", "Nifty average", "Nifty worst 1%", "S&P average", "S&P worst 1%"], [
  ["Real trades", "−12.0%", "−17.3%", "−11.4%", "−17.2%"],
  ["Random walk", "−10.0%", "−10.5%", "−10.0%", "−10.6%"],
  ["Brownian motion", "−10.6%", "−12.1%", "−10.7%", "−12.6%"],
  ["Jump-diffusion", "**−11.2%**", "**−17.4%**", "**−11.1%**", "**−17.0%**"],
], [2126, 1725, 1725, 1725, 1725]));

// ---------- Betting game
C.push(H1("11. Betting Game"));
C.push(H2("11.1 How much to bet"));
C.push(P("The Kelly criterion applied to each month's model recommends betting, on average, a little over twice one's wealth, but the recommendation moves a great deal (Figure 5). On the S&P 500 it swings from nothing after the falls of 2000–2002 and 2008 to the leverage limit during the rises of the late 1990s and 2010s. The shaded band shows why: changing the estimated average return by one standard error moves the bet across almost its whole range, and five years of data cannot measure that return precisely. The jump-diffusion recommends somewhat smaller bets, because it expects occasional large losses."));
C.push(...FIG("q4_fstar_series_spx", "The Kelly bet recommended by each month's model (S&P 500). Shaded: the Brownian-motion recommendation if the average return were one standard error higher or lower. Dotted: the three-times-wealth limit used in the betting game.", 560));

C.push(H2("11.2 How the game works"));
C.push(P("The game turns the barrier trade into a repeated bet. A trader starts with ₹100 (or $100) and plays 100 trades one after another. Before each trade they fit the model to the previous five years of prices and use it to choose a fraction f of their current wealth to bet, at most three times their wealth; the trade then runs until it reaches +20% or −10%, or the two-year limit."));
C.push(P("Wealth carries over from trade to trade. A trader with ₹100 who always bets half of their wealth puts ₹50 into the first trade; a 20% gain adds ₹10, so the next trade starts with ₹110 and bets ₹55; if that trade is stopped out at −10%, the trader loses ₹5.50 and continues with ₹104.50. Because each trade multiplies wealth, the final result is decided by the long-run growth rate, the average rate at which these multiplications compound."));
C.push(...TABLE("The betting rules compared.", ["Rule", "How the bet is chosen before each trade"], [
  ["Kelly (Brownian motion)", "the model's Kelly fraction, or a multiple of it from a quarter to double; Half Kelly bets half of it"],
  ["Kelly (jump-diffusion)", "the same, using the jump-diffusion model"],
  ["Empirical Kelly", "the fraction that would have worked best on the trades of the last five years, without any model"],
  ["Adjusted Half Kelly", "Half Kelly, made smaller after volatile periods, after losses, or when the estimated return is uncertain"],
  ["Fixed 50% / Flat ₹50", "always half of current wealth / always ₹50"],
  ["Martingale", "double the bet after every loss and return to the starting bet after a win"],
  ["Buy and hold", "invest everything at the start and never sell"],
], [3026, 6000], ["left", "left"]));
C.push(P("Each rule plays 2,000 careers on the same price paths, so differences come from the rules rather than from luck. The rules are compared by how fast their wealth grows and by their drawdown, the largest fall from a previous peak, which shows how painful a rule is to follow."));
C.push(H2("11.3 Results"));
C.push(P("In model-generated markets the true best bet is known, so we can see how far the trader's estimate strays from it (Figure 6). The estimates spread widely, from no bet at all to three times too much. Because growth falls steeply beyond its peak, over-bets cost far more than under-bets save, so following the estimate in full grows wealth noticeably more slowly than knowing the truth."));
C.push(...FIG("q5_misestimation_nifty", "Top: the true long-run growth for each size of bet in markets generated by each model (Nifty parameters). Bottom: the bets the trader actually chose from five years of data before each trade.", 560));
C.push(P("Figure 7 compares each strategy with half-Kelly betting in the market rebuilt from real returns, alongside the depth of temporary losses; Table 5 converts the results into money."));
C.push(...FIG("q5_leaderboard_nifty", "Market rebuilt from real Nifty returns, 2,000 careers of 100 trades: long-run growth of each strategy compared with half-Kelly betting (left) and the depth of temporary losses (right).", 430));
C.push(...TABLE("What ₹100 or $100 becomes after 100 trades in the rebuilt market (typical career), and on the actual history.", ["Strategy", "Nifty, 100 trades", "Nifty, actual", "S&P, 100 trades", "S&P, actual"], [
  ["Buy and hold", "₹19,990", "₹284", "$24,782", "$1,401"],
  ["Half Kelly, jump-diffusion", "**₹6,026**", "₹191", "**$4,238**", "$638"],
  ["Half Kelly, Brownian motion", "₹5,131", "₹190", "$3,424", "$745"],
  ["Full Kelly, Brownian motion", "₹2,075", "₹182", "$2,478", "$146"],
  ["Double Kelly", "₹1,088", "₹264", "$1,741", "$78"],
  ["Cash only", "₹2,190", "≈ ₹160", "$674", "≈ $215"],
], [2826, 1550, 1550, 1550, 1550]));
C.push(P("Five patterns hold in every kind of market we tried."));
C.push(B("**Betting less than Kelly works better.** Half Kelly beats full Kelly in every market, and betting more than Kelly is worse still, with far deeper losses."));
C.push(B("**The jump-diffusion gives the best bet sizes.** Half Kelly driven by the jump-diffusion beats the same rule driven by Brownian motion everywhere, because it allows for the large losses real stops suffer."));
C.push(B("**Models help.** Model-based Kelly beats Kelly based directly on past trade outcomes, so the models make better use of the same five years of data."));
C.push(B("**Simple refinements do not help.** Scaling the bet by recent volatility, cutting it after losses or shrinking the estimated return did not beat plain half Kelly, and doubling after losses did worse in every market."));
C.push(B("**Holding the index grows fastest.** Buy-and-hold avoids costs and noisy estimates, and for the Nifty its constant exposure sits close to the true best bet. The price is deep temporary losses: in a typical career it falls more than half below its peak at some point."));
C.push(P("The mechanics behind two of the heuristics are worth spelling out. Martingale looks attractive because one win after a run of losses recovers them, but losses in real markets come in clusters, so it reaches its largest bets exactly when the next loss is most likely. The flat ₹50 bet is the safest rule, with the smallest drawdowns, but because the bet never grows with the account it compounds slowly and ends with little more than cash."));
C.push(...FIG("q5_worlds_spx", "Long-run growth of every strategy in each kind of market (S&P 500). Coloured dots: markets generated by the models; black dots: market rebuilt from real returns; crosses: the actual 1994–2025 history.", 330));
C.push(P("The markets rank the strategies in much the same order (Figure 8), so the models are good enough for comparing strategies despite missing clustering. The rebuilt real market is harsher on heavy borrowing, and the actual S&P 500 history harsher still: the two bear markets of the 2000s left the most aggressive strategies with almost nothing after thirty years."));

// ---------- Validation
C.push(H1("12. Model Validation"));
C.push(P("The models were validated at three levels. Mathematically, every simulation matched its exact solution. On data, every forecast used only information available at the time and was judged against what followed; automated tests confirm that changing a path's future never changes an earlier decision. For decisions, the betting rules were tested not only in model-generated markets, which would favour the models, but also in a market rebuilt from real returns and on the actual history."));
C.push(P("The evidence supports part of the hypothesis. The models, especially the jump-diffusion, describe the size of daily moves and the losses on stopped trades, and forecast the trade better than no model. They do not describe how volatility changes over time, cannot be told apart on the trade's outcome, and imply bet sizes that are useful only when scaled down."));

// ---------- Discussion
C.push(H1("13. Discussion"));
C.push(P("The jump-diffusion's advantage has a clear economic meaning: its heavier tails make it expect occasional large losses, so it predicts realistic stop-loss gaps and smaller bets, which is what pays off in the betting game. What all three models miss is that volatility is itself random and persistent, which shows up as too many very short and very long trades and as deeper losses in the rebuilt real market than in model-generated ones."));
C.push(P("The most important lesson concerns the inputs rather than the models. The chance of winning and the Kelly bet both depend mainly on the average return, the one quantity five years of data measure poorly, so a carefully computed optimal bet can be far from the truth. Betting a fraction of it works because over-betting does more damage than under-betting by the same amount, the practical case for fractional Kelly made by MacLean, Thorp and Ziemba [15]; the strong showing of buy-and-hold makes the same point from the other side."));

// ---------- Limitations
C.push(H1("14. Limitations and Extensions"));
C.push(B("The Nifty 50 test period covers nine years, so many comparisons on that index are inconclusive, and each actual price history is a single sample."));
C.push(B("Five years of data suit the estimation of volatility, but leave the average return uncertain. Estimating the average return from all available history, while keeping volatility on recent data, may give steadier bets."));
C.push(B("The model-generated markets keep their parameters fixed, and the rebuilt market only reuses past days in short blocks, so long regime changes and crashes larger than any in history are under-represented."));
C.push(B("Interest rates are held constant, and trading costs, borrowing costs and the leverage limit are assumptions."));
C.push(P("Natural extensions are a GARCH model, which lets volatility cluster and in a preliminary test fitted daily returns better than both models used here; markets whose parameters change over time; a longer history for the average return; and intraday prices for the stop-loss."));

// ---------- Conclusion
C.push(H1("15. Conclusion"));
C.push(P("A take-profit and stop-loss trade is a gambler's-ruin problem that stochastic-process theory solves exactly once daily price checks are modelled. On real markets, the jump-diffusion describes the size of daily moves and the losses on stopped trades far better than Brownian motion or the random walk, and all three models forecast the trade better than no model, but none captures volatility clustering. For bet sizing the models must be used with caution: the optimal bet depends on an average return that cannot be measured precisely, betting half of it is consistently better than betting all of it, and the jump-diffusion gives the best bet sizes. Holding the index grows wealth faster still, for a trader who can live through losing more than half of it at times."));

// ---------- References
C.push(H1("References"));
const refs = [
  "W. Feller, *An Introduction to Probability Theory and Its Applications*, Vol. 1, 3rd ed. Wiley, 1968.",
  "S. Karlin and H. M. Taylor, *A First Course in Stochastic Processes*, 2nd ed. Academic Press, 1975.",
  "F. Black and M. Scholes, \"The pricing of options and corporate liabilities,\" *Journal of Political Economy*, 81(3), 637–654, 1973.",
  "R. C. Merton, \"Option pricing when underlying stock returns are discontinuous,\" *Journal of Financial Economics*, 3, 125–144, 1976.",
  "M. Broadie, P. Glasserman and S. Kou, \"A continuity correction for discrete barrier options,\" *Mathematical Finance*, 7(4), 325–349, 1997.",
  "R. Cont, \"Empirical properties of asset returns: stylized facts and statistical issues,\" *Quantitative Finance*, 1, 223–236, 2001.",
  "G. M. Ljung and G. E. P. Box, \"On a measure of lack of fit in time series models,\" *Biometrika*, 65(2), 297–303, 1978.",
  "F. X. Diebold, T. A. Gunther and A. S. Tay, \"Evaluating density forecasts with applications to financial risk management,\" *International Economic Review*, 39(4), 863–883, 1998.",
  "T. Gneiting and A. E. Raftery, \"Strictly proper scoring rules, prediction, and estimation,\" *Journal of the American Statistical Association*, 102, 359–378, 2007.",
  "G. W. Brier, \"Verification of forecasts expressed in terms of probability,\" *Monthly Weather Review*, 78(1), 1–3, 1950.",
  "H. R. Künsch, \"The jackknife and the bootstrap for general stationary observations,\" *Annals of Statistics*, 17(3), 1217–1241, 1989.",
  "D. N. Politis and J. P. Romano, \"The stationary bootstrap,\" *Journal of the American Statistical Association*, 89, 1303–1313, 1994.",
  "J. L. Kelly, \"A new interpretation of information rate,\" *Bell System Technical Journal*, 35(4), 917–926, 1956.",
  "E. O. Thorp, \"The Kelly criterion in blackjack, sports betting and the stock market,\" in *Handbook of Asset and Liability Management*, Vol. 1, Elsevier, 2006.",
  "L. C. MacLean, E. O. Thorp and W. T. Ziemba (eds.), *The Kelly Capital Growth Investment Criterion*. World Scientific, 2011.",
];
refs.forEach((r, i) => C.push(new Paragraph({ children: runs(`[${i + 1}] ${r}`, { size: 17 }), spacing: { after: 30 },
  indent: { left: 400, hanging: 400 } })));

// =============================================================== document
const doc = new Document({
  creator: "Barrier first-passage study",
  title: "Stochastic Models of a Take-Profit / Stop-Loss Trade and a Kelly Betting Game",
  styles: {
    default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { line: 247 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "000000" },
        paragraph: { spacing: { before: 260, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, italics: true, color: "000000" },
        paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "666666" })] })] }) },
    children: C,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
