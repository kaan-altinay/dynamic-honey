# Agentic Pipeline Integration into Tanner/Snare

## What is it?
Integrated an agentic live-target generation pipeline into the `snare` + `tanner` honeypot codebase so missed HTTP paths can trigger generation of a small, believable static resource bundle and then be persisted for later serving.

## What has been implemented

### 1. TANNER: generator upgrade
- Replaced the old single-page `generate_page(...)` contract with a bundle-returning `generate_bundle(...)` contract. The bundle in question is a collection of resources that will be served to make the served page react realistically.
- Replaced the `local_qwen` path with a new provider-agnostic agentic generator under `tanner/tanner/generator/agentic/`. (So I can switch between local and OpenAI providers for the model calls)
- Added:
  - Pydantic models for the `generation request`, `expert spec`, `planned/draft/generated artifact`, `generated bundle`, and `review decision` objects.
  - provider/model factory via LangChain: This is a framework that makes chaining LLM interactions easier, LangGraph allows the model workflow:
    - `Expert`: infers intent, outputs a (JSON) spec. Can search the internet.
    - `Design`: Uses expert spec (and command execution, but this is iffy at the moment) to design a pack of artifacts.
    - Swarm of `Coder`s: slaves to generate artifacts requested by `Design`
    - `Review`: Checks the final bundle. Sends back to design if it does not approve (for a capped number of times)
    - `Finalize`: Decides on whether final bundle is worth sending back to Snare, or if its filled with too many fallbacks and should not be sent.
  - deterministic renderers and validators for now, working on removing these to keep it fully agentic besides in the finalize part.
- Updated TANNER job payloads from one artifact to a reviewed bundle:
  - `primary_path`
  - `artifacts[]`: crust of the pack
  - `review_summary`: what review agent said about the pack
  - `used_fallback`: Bool

### 2. SNARE: bundle persistence
- Updated SNARE polling and persistence to consume bundle-shaped TANNER job results.
- SNARE now:
  - decodes every generated artifact in the bundle
  - writes each content blob to disk
  - atomically updates `meta.json` for all generated bundle paths
  - serves later requests for adjacent generated resources from the persisted bundle

### 3. Config and backend selection
- Added provider-agnostic generator config in `tanner/tanner/data/config.yaml`.
- TANNER now selects the new `AgenticBundleGenerator` when `GENERATOR.backend=agentic`.
- Added dependencies for LangChain / LangGraph / provider integrations.

## Current technical status
- The core bundle-generation architecture is integrated into the honeypot codebase.
- The SNARE / TANNER communication contract now supports multi-artifact bundles instead of a single, weak generated page. Yet to see if this is stronger but I am optimistic.
- The deterministic fallback path is expanded to be a fallback per class of requested resource (**CMS probe**, **config theft** etc.) so fallback families can be added without growing hardcoded branching logic.

## Current diagnosis after some testing (I started this this morning)
- I tried to test with the  local `llama.cpp` / Qwen server as it is also an OpenAI-compatible backend, but I have to tune the response format type for that to work. Even when it does, it should not have enough generation power to abide by the schemas in my workflow, so I suspended work on the local model for now.

- In the `OpenAI` tests, I hit a usage limit even though I shouldn't yet. I want to fix this first so I can see how many tokens is used, so that I can estimate what size of a local model would be needed to generate results of the same caliber (and thus how much `GPU` power I would need to test that).

## Summary
The architectural integration is fine, I validated that with tests and the pipeline flows with the default values when the models cannot be reached (`Expert` -> `Design` -> [`Coder`] -> `Review` -> `Finalize`). I now need to get it to run on `OpenAI` models and see: 
1. Are the generated artifacts as good as I expect them to be? 
2. What sort of computing power do we need to get similar results from a local model (of how many parameters)?

I should have answers to these two questions by the time we manage to meet this week.
