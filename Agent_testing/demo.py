"""Agent Simulator with JSON Schema Validation

Interactive simulator that reads user input from the keyboard, classifies the
intent (weather, stock, or other) using ChatGPT when an OpenAI API key is set
and available, otherwise falls back to a simple rule-based classifier.

The ChatGPT classifier is asked to return a JSON object with fields:
  {
    "intent": "weather|stock|other",
    "tools": ["tool1", "tool2"],
    "params": { ... optional suggested parameters ... }
  }

After classifying, the simulator builds a JSON payload that simulates a REST
call to a webserver, validating against JSON schemas, and logs the payload
and observability metrics.

Usage:
  python demo.py

Optional environment variables:
  OPENAI_API_KEY - if set the simulator will call OpenAI's chat completion API
  SIM_ENDPOINT - if set, the simulator will POST the simulated payload to this URL
  MODEL - optional (default: gpt-3.5-turbo)

Commands inside the simulator:
  stats   - show observability metrics
  exit    - quit
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from jsonschema import validate, ValidationError

# Paths
SCHEMAS_DIR = pathlib.Path(__file__).parent / "schemas"
WEATHER_SCHEMA = SCHEMAS_DIR / "weather.schema.json"
STOCK_SCHEMA = SCHEMAS_DIR / "stock.schema.json"
COMMON_SCHEMA = SCHEMAS_DIR / "common.schema.json"

# Logging config
logger = logging.getLogger("AgentSimulator")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)


class Observability:
    def __init__(self) -> None:
        self.total_requests = 0
        self.intent_counts: Dict[str, int] = defaultdict(int)
        self.total_classify_time = 0.0
        self.validation_errors = 0

    def record(self, intent: str, classify_time: float) -> None:
        self.total_requests += 1
        self.intent_counts[intent] += 1
        self.total_classify_time += classify_time

    def record_validation_error(self) -> None:
        self.validation_errors += 1

    def snapshot(self) -> Dict[str, Any]:
        avg = (self.total_classify_time / self.total_requests) if self.total_requests else 0.0
        return {
            "total_requests": self.total_requests,
            "intent_counts": dict(self.intent_counts),
            "avg_classify_time_s": round(avg, 4),
            "validation_errors": self.validation_errors,
        }


class AgentSimulator:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("MODEL", "gpt-3.5-turbo")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.sim_endpoint = os.getenv("SIM_ENDPOINT")
        self.obs = Observability()
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load and store JSON schemas for validation."""
        with open(WEATHER_SCHEMA, encoding='utf-8') as f:
            self.weather_schema = json.load(f)
        with open(STOCK_SCHEMA, encoding='utf-8') as f:
            self.stock_schema = json.load(f)
        with open(COMMON_SCHEMA, encoding='utf-8') as f:
            self.common_schema = json.load(f)

    def classify_with_openai(self, user_text: str) -> Dict[str, Any]:
        """Call OpenAI Chat Completions (via REST HTTP) and expect a JSON-only reply.

        Returns a dict with keys: intent, tools, params.
        """
        if not self.openai_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        prompt = (
            "You are an assistant that must classify a user's request into one of "
            "these intents: 'weather', 'stock', or 'other'.\n"
            "You must also propose two simple tool names appropriate for this task.\n"
            "Return STRICTLY a single JSON object with keys: intent, tools, params.\n"
            "- intent: one of 'weather','stock','other'\n"
            "- tools: array of two short tool names (strings)\n"
            "- params: a small object with any extracted parameters, e.g. {\"location\":\"Paris\"}\n"
            "Do not include any explanation or extra text.\n\n"
            f"User: {user_text}\n"
        )

        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only a JSON object as described."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 200,
            "temperature": 0,
        }

        url = "https://api.openai.com/v1/chat/completions"

        logger.debug("Calling OpenAI chat completions (model=%s)", self.model)
        r = requests.post(url, headers=headers, json=body, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Try to extract the assistant text
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            # older or alternative shape
            content = data["choices"][0].get("text", "")

        content = content.strip()
        logger.debug("OpenAI raw response: %s", content[:400])

        # Find the first JSON object in the content
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise ValueError("OpenAI response did not contain JSON")

        json_text = m.group(0)
        parsed = json.loads(json_text)
        return parsed

    def classify_rule_based(self, user_text: str) -> Dict[str, Any]:
        text = user_text.lower()
        params = {}
        # Detect weather
        if any(w in text for w in ("weather", "forecast", "temperature", "rain", "sunny")):
            # try to extract a location
            loc = self._extract_location(user_text)
            if loc:
                params["location"] = loc
            return {"intent": "weather", "tools": ["weather_api", "geocode"], "params": params}

        # Detect stock
        stock_words = ("stock", "price", "ticker", "quote", "shares", "actiuni", "pret", "valoare")
        if any(w in text for w in stock_words):
            ticker = self._extract_ticker(user_text)
            if ticker:
                params["ticker"] = ticker
                return {"intent": "stock", "tools": ["stock_price_api", "company_info"], "params": params}
            # If we detected stock intent but no ticker, try again with company name
            ticker = self._extract_ticker(text)
            if ticker:
                params["ticker"] = ticker
            return {"intent": "stock", "tools": ["stock_price_api", "company_info"], "params": params}

        # fallback
        return {"intent": "other", "tools": ["web_search", "no_op"], "params": {}}

    def _extract_location(self, text: str) -> Optional[str]:
        # simple heuristic: look for 'in X' or capitalized token sequences
        m = re.search(r"in ([A-Za-z ,]+)$", text)
        if m:
            return m.group(1).strip()
        # try find a capitalized word
        m2 = re.search(r"\b([A-Z][a-z]+)\b", text)
        return m2.group(1) if m2 else None

    def _extract_ticker(self, text: str) -> Optional[str]:
        # look for $AAPL or upper-case sequences of 1-5 letters
        m = re.search(r"\$([A-Za-z]{1,5})", text)
        if m:
            return m.group(1).upper()
        m2 = re.search(r"\b([A-Z]{1,5})\b", text)
        if m2 and len(m2.group(1)) <= 5 and text.count(m2.group(1)):
            return m2.group(1).upper()
        # Map common company names to tickers
        company_map = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "facebook": "META",
            "netflix": "NFLX"
        }
        text_lower = text.lower()
        for company, ticker in company_map.items():
            if company in text_lower:
                return ticker
        return None

    def simulate_rest_payload(self, classification: Dict[str, Any], user_text: str) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "intent": classification.get("intent"),
            "tools": classification.get("tools"),
            "params": classification.get("params"),
            "original_text": user_text,
        }
        
        try:
            # Validate against common schema first
            validate(instance=payload, schema=self.common_schema)
            
            # Additional validation based on intent
            intent = classification.get("intent")
            if intent == "weather":
                validate(instance=payload["params"], schema=self.weather_schema)
            elif intent == "stock":
                validate(instance=payload["params"], schema=self.stock_schema)
            
        except ValidationError as e:
            logger.error("Schema validation failed: %s", e)
            self.obs.record_validation_error()
            # Add validation error info to payload
            payload["validation_error"] = str(e)
            
        return payload

    def maybe_send(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[requests.Response]]:
        if not self.sim_endpoint:
            logger.debug("SIM_ENDPOINT not set; skipping POST")
            return False, None
        try:
            r = requests.post(self.sim_endpoint, json=payload, timeout=10)
            logger.info("Posted simulated payload to %s (status=%s)", self.sim_endpoint, r.status_code)
            return True, r
        except Exception as e:
            logger.exception("Failed to POST simulated payload: %s", e)
            return False, None

    def handle_input(self, user_text: str) -> Dict[str, Any]:
        start = time.time()
        classification = None
        used_openai = bool(self.openai_key)
        try:
            if used_openai:
                try:
                    classification = self.classify_with_openai(user_text)
                except Exception:
                    logger.exception("OpenAI classification failed, falling back to rule-based")
                    classification = self.classify_rule_based(user_text)
                    used_openai = False
            else:
                classification = self.classify_rule_based(user_text)
        finally:
            classify_time = time.time() - start
            self.obs.record(classification.get("intent", "other"), classify_time)

        payload = self.simulate_rest_payload(classification, user_text)
        logger.info("Simulated Request ID=%s intent=%s tools=%s", payload["request_id"], payload["intent"], payload["tools"])
        logger.debug("Payload: %s", json.dumps(payload, indent=2)[:1000])

        sent, resp = self.maybe_send(payload)
        return {"payload": payload, "sent": sent, "response": resp, "used_openai": used_openai}


def main_loop():
    sim = AgentSimulator()
    print("Agent Simulator. Type a request (or 'stats' / 'exit').")
    while True:
        try:
            txt = input("-> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not txt:
            continue
        if txt.lower() in ("exit", "quit"):
            break
        if txt.lower() == "stats":
            print("Observability:", json.dumps(sim.obs.snapshot(), indent=2))
            continue

        result = sim.handle_input(txt)
        print(json.dumps(result["payload"], indent=2))
        if result["sent"]:
            print("(payload posted to", sim.sim_endpoint, ")")


if __name__ == "__main__":
    main_loop()
