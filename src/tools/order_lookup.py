"""
Order lookup tool for the Aster & Row support agent.

Design principles (see PRD Section 2 for the full data analysis this is
based on):

- The allowlist is enforced at THIS layer, before anything reaches the LLM.
  Internal fields (customer PII, risk_score, warehouse_note, support_tags)
  are never read out of the source record in the first place -- the safe
  payload is built field-by-field from an explicit allowlist, not produced
  by taking the full record and trying to strip/redact the bad parts after
  the fact. Redact-after-the-fact risks a forgotten field; build-from-
  allowlist cannot leak a field nobody explicitly listed.
- `status` is authoritative. A stale carrier/tracking/estimated_delivery
  value left over from before a cancellation/return is suppressed based on
  status, never reported as if the order is still in transit.
- Order IDs are normalized (trimmed, uppercased) before lookup. A value
  that doesn't match after normalization is reported as not found -- we
  never guess a nearby/similar order ID.
- The dataset's own `snapshot_at` is used as "now" for time-based logic
  (the 30-minute cancellation window), not the real system clock. This
  keeps evaluation deterministic regardless of when the eval suite runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


CANCELLATION_WINDOW_MINUTES = 30

# Statuses where a previously-recorded delivery estimate is stale and must
# not be reported as if the order is still in transit.
TERMINAL_NON_DELIVERY_STATUSES = {"cancelled", "returned"}


@dataclass
class OrderLookupResult:
    found: bool
    order_id: str
    data: dict | None = None
    error: str | None = None  # "missing_id" | "not_found"
    can_still_cancel: bool | None = None

    def to_tool_payload(self) -> dict:
        """The exact dict that is safe to place in the model's context.
        Nothing outside of what this method returns should ever reach the
        LLM."""
        if not self.found:
            return {"found": False, "order_id": self.order_id, "error": self.error}

        d = self.data
        payload = {
            "found": True,
            "order_id": d["order_id"],
            "membership_tier": d["membership_tier"],
            "items": [
                {"name": i["name"], "quantity": i["quantity"], "final_sale": i["final_sale"]}
                for i in d["items"]
            ],
            "placed_at": d["placed_at"],
            "status": d["status"],
            "status_updated_at": d["status_updated_at"],
            "shipped_at": d["shipped_at"],
            "delivered_at": d["delivered_at"],
            "carrier": d["carrier"],
            "tracking_number": d["tracking_number"],
            "customer_safe_message": d["customer_safe_message"],
            "can_still_cancel": self.can_still_cancel,
        }

        # Suppress a stale delivery estimate on cancelled/returned orders.
        if d["status"] in TERMINAL_NON_DELIVERY_STATUSES:
            payload["estimated_delivery"] = None
        else:
            payload["estimated_delivery"] = d["estimated_delivery"]

        return payload


class OrderLookupTool:
    def __init__(self, orders_path: str | Path):
        raw = json.loads(Path(orders_path).read_text(encoding="utf-8"))
        self.snapshot_at = datetime.fromisoformat(raw["snapshot_at"].replace("Z", "+00:00"))
        self._orders_by_id = {o["order_id"]: o for o in raw["orders"]}

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        return raw_id.strip().upper()

    def lookup(self, order_id: str | None) -> OrderLookupResult:
        if not order_id or not order_id.strip():
            return OrderLookupResult(found=False, order_id="", error="missing_id")

        normalized = self.normalize_order_id(order_id)
        order = self._orders_by_id.get(normalized)

        if order is None:
            return OrderLookupResult(found=False, order_id=normalized, error="not_found")

        return OrderLookupResult(
            found=True,
            order_id=normalized,
            data=order,
            can_still_cancel=self._can_still_cancel(order),
        )

    def _can_still_cancel(self, order: dict) -> bool:
        if order["status"] != "pending":
            return False
        placed_at = datetime.fromisoformat(order["placed_at"].replace("Z", "+00:00"))
        return (self.snapshot_at - placed_at) <= timedelta(minutes=CANCELLATION_WINDOW_MINUTES)


# Gemini function-calling schema for this tool. Wired up in llm_client.py
# in Phase 3.
ORDER_LOOKUP_TOOL_SCHEMA = {
    "name": "order_lookup",
    "description": (
        "Look up the current status of a customer's order by order ID. "
        "Returns status, shipping info, and a customer-safe summary. "
        "Never returns customer PII or internal notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID, e.g. 'ORD-1007'. Case and surrounding whitespace do not matter.",
            }
        },
        "required": ["order_id"],
    },
}


# Fields/strings that must never appear anywhere in a tool payload, no
# matter which order or code path produced it. Used both as an inline
# self-check here and importable by the eval suite in Phase 7.
FORBIDDEN_LEAK_MARKERS = [
    "internal",
    "risk_score",
    "warehouse_note",
    "support_tags",
    "email",
    "shipping_address",
    "customer.name",
]


if __name__ == "__main__":
    import sys

    orders_path = sys.argv[1] if len(sys.argv) > 1 else "../data/orders.json"
    tool = OrderLookupTool(orders_path)

    test_ids = [
        "ORD-1007",    # normal shipped order
        " ord-1007 ",  # normalization test (lowercase + whitespace)
        "ORD-1004",    # cancelled, stale estimated_delivery in source data
        "ORD-1008",    # returned
        "ORD-1010",    # exception status, no ETA
        "ORD-1011",    # shipped, no ETA
        "ORD-1001",    # pending, inside the 30-minute cancellation window
        "ORD-1012",    # processing, cancellation window already closed
        "ORD-9999",    # unknown
        "",            # missing
    ]

    all_passed = True
    for tid in test_ids:
        result = tool.lookup(tid)
        payload = result.to_tool_payload()
        print(f"lookup({tid!r}) ->")
        print(json.dumps(payload, indent=2))

        serialized = json.dumps(payload)
        for marker in FORBIDDEN_LEAK_MARKERS:
            if marker in serialized:
                print(f"  !!! LEAK DETECTED: {marker!r} found in payload for {tid!r}")
                all_passed = False
        print()

    print("ALL LEAK CHECKS PASSED" if all_passed else "LEAK CHECKS FAILED -- see above")