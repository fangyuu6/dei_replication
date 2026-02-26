# Analysis & Improvement Plan: Data Diversity and LLM Bias

## Context
User observed two problems:
1. The database contains repetitive/similar bills
2. Feasibility scores may be biased by LLM's prior knowledge of existing literature

---

## Problem 1: Why Bills Are Repetitive

### Root Cause Analysis

**A. Hardcoded URL lists are the main culprit (when using URL mode)**
- `input/bill_urls_test.txt` and `input/bill_urls_aer_run.txt` contain ~58 hand-picked "classic" policy bills
- These are overwhelmingly well-known labor economics natural experiments (CA AB5, TX SB8, minimum wage bills, paid leave, gun control)
- Running from these lists will always produce the same results

**B. Labor economics keyword filter narrows the funnel**
- When using the batch pipeline (`batch_collect.py`), a filter with 40+ regex patterns (wage, employment, union, hiring, discrimination, etc.) filters collected bills
- This inherently biases toward "classic" labor econ topics that academics already study
- The same policy areas (minimum wage, paid leave, gig worker classification, gun control) dominate because those are the keywords

**C. State crawler coverage is limited**
- Only 4 states have dedicated crawlers: FL, TX, CA, NY
- These are large, high-profile states whose legislation already gets academic attention
- Smaller states with equally interesting but less-known policies are underrepresented

**D. Congress.gov dominance**
- Federal bills are high-profile and well-known by definition
- Many are "messaging bills" that never pass, reducing useful variety

### Recommendations for More Diverse Data

1. **Expand beyond labor economics keywords** — run with `--no-filter-labor` or add filters for:
   - Health policy (Medicaid expansion, telehealth, drug pricing)
   - Environmental (emissions standards, renewable mandates, PFAS regulations)
   - Housing (zoning reform, rent control, eviction moratoriums)
   - Education (school choice, funding formulas, testing requirements)
   - Criminal justice (bail reform, sentencing changes, police reform)
   - Technology (privacy laws, AI regulation, broadband subsidies)

2. **Add new data sources:**
   - **NCSL (National Conference of State Legislatures)** — curated policy trackers by topic
   - **LegiScan API** — covers all 50 states + DC + US Congress, 2.5M+ bills
   - **Legiscan bulk datasets** — historical state legislation going back to 2010
   - **EU/UK legislation** — EUR-Lex, legislation.gov.uk for international quasi-experiments
   - **Municipal ordinances** — Municode, American Legal Publishing for city-level policy variation
   - **Canadian legislation** — LEGISinfo (federal), provincial legislature sites
   - **World Bank / ILO policy databases** — developing country policy changes

3. **Diversify the URL seed lists** — instead of hand-picked "classics", use:
   - Random sampling from Open States bulk data
   - Topic-stratified sampling across policy areas
   - Deliberate inclusion of small/medium states

---

## Problem 2: LLM Feasibility Bias (Training Data Contamination)

### The user's intuition is correct — this is a real and significant bias

**The mechanism:**
1. LLM (GPT-4o-mini) was trained on academic papers, NBER working papers, economics blogs, etc.
2. For **well-studied policies** (e.g., minimum wage, ACA Medicaid expansion, CA AB5):
   - The model "knows" these work as natural experiments because it has read the papers
   - It can articulate clear treatment/control groups, cite specific datasets, describe identification strategies
   - → Higher scores on identification, data feasibility, and treatment intensity
3. For **unstudied policies** (e.g., a novel state occupational licensing reform):
   - The model has no academic "template" to draw from
   - It struggles to articulate a clear identification strategy because no one has done it yet
   - → Lower scores, especially on identification and data feasibility
   - The model may label it as "no credible design" when in reality a clever researcher could find one

**This creates a self-reinforcing loop:**
- Well-studied → LLM says feasible → gets flagged as "Pursue"
- Novel/unstudied → LLM says infeasible → gets flagged as "Drop"
- Result: the tool recommends researching what's already been researched

**Evidence in the prompt design:**
- Phase 4 explicitly asks the LLM to assess "literature landscape" from its own knowledge
- The `known_similar_studies` field asks the model to list papers it knows about
- `literature_saturation` scoring uses the model's recall of existing literature as a proxy for novelty
- But the model's knowledge of literature is biased toward famous/well-cited papers

### Proposed Mitigations

1. **Separate "design feasibility" from "literature familiarity"**
   - Score identification strategy purely based on bill text (does it have a threshold? a control group?)
   - Do NOT let literature knowledge influence the design score
   - Add explicit prompt instruction: "Score identification based ONLY on the bill's text and mechanics, NOT on whether you've seen similar studies"

2. **Invert the literature signal**
   - Currently: crowded literature → penalty, blue_ocean → bonus (in `feasibility_scorer.py`)
   - Problem: the LLM's assessment of "blue_ocean" is unreliable because absence of knowledge ≠ absence of papers
   - Better: use the external OpenAlex/Crossref check (`literature_checker.py`) as the ONLY literature signal, not the LLM's internal knowledge

3. **Add a "novelty bonus" instead of penalizing unknowns**
   - If OpenAlex finds < 3 papers on a topic AND the design score is moderate (>40), boost the score
   - This rewards genuinely novel opportunities rather than penalizing them

4. **Calibration prompt**
   - Add to system prompt: "A policy you have never seen studied before may be MORE valuable as a research opportunity, not less. Do not penalize novelty."

5. **Ablation test**
   - Run the same bills through with and without literature assessment
   - Compare scores to quantify the bias magnitude

---

## Proposed Implementation

### Phase A: Fix LLM prompt bias (high impact, low effort)
- Modify `extraction/prompts_aer.py` to add anti-contamination instructions
- Separate design scoring from literature familiarity
- Add novelty calibration language

### Phase B: Expand data sources (medium effort)
- Add LegiScan API integration as a new ingestion source
- Broaden keyword filters or add multi-topic mode
- Add topic-stratified random sampling

### Phase C: Fix literature scoring pipeline (medium effort)
- Remove LLM-based literature saturation from composite score
- Use only external API (OpenAlex) for literature assessment
- Add novelty bonus logic

### Verification
- Re-run a sample batch comparing old vs new prompts
- Check if previously "dropped" novel policies get higher scores
- Verify that well-known policies don't get artificially boosted
