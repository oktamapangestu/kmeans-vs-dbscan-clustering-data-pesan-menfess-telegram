import argparse
import asyncio
import csv
import os
import sys

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Message

from project_paths import DATA_RAW_DIR, ensure_parent_dir


FIELDNAMES = [
    "id",
    "date",
    "text",
]


def msg_to_row(m: Message) -> dict:
    return {
        "id": m.id,
        "date": m.date.isoformat() if m.date else "",
        "text": m.message or "",
    }


async def main() -> int:
    # Load environment variables from .env (if present) before reading os.getenv defaults.
    load_dotenv(override=False)

    ap = argparse.ArgumentParser(description="Grab all Telegram channel posts to CSV using Telethon")
    ap.add_argument("--api-id", type=int, default=int(os.getenv("TG_API_ID", "0")))
    ap.add_argument("--api-hash", type=str, default=os.getenv("TG_API_HASH", ""))
    ap.add_argument("--session", type=str, default=os.getenv("TG_SESSION", "tg_grab"))
    ap.add_argument("--channel", type=str, required=True, help="e.g. @mychannel or https://t.me/mychannel")
    ap.add_argument("--out", type=str, default=str(DATA_RAW_DIR / "export.csv"))
    ap.add_argument("--resume-from-id", type=int, default=0, help="skip messages with id <= this")
    ap.add_argument("--reverse", action="store_true", help="oldest->newest")
    ap.add_argument(
        "--limit",
        "--max-data",
        type=int,
        default=0,
        help="max messages to fetch (0 = no limit)",
    )
    ap.add_argument("--progress-every", type=int, default=250, help="0 = no progress output")
    args = ap.parse_args()

    if not args.api_id or not args.api_hash:
        print(
            "Missing api credentials. Provide --api-id/--api-hash or env TG_API_ID/TG_API_HASH.",
            file=sys.stderr,
        )
        return 2

    ensure_parent_dir(args.out)

    need_header = (not os.path.exists(args.out)) or (os.path.getsize(args.out) == 0)

    client = TelegramClient(args.session, args.api_id, args.api_hash)
    await client.start()

    entity = await client.get_entity(args.channel)

    written = 0
    last_id = args.resume_from_id

    with open(args.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        if need_header:
            w.writeheader()

        while True:
            try:
                it = client.iter_messages(
                    entity,
                    limit=(args.limit if args.limit > 0 else None),
                    reverse=args.reverse,
                    min_id=args.resume_from_id,  # strict: > min_id
                )

                async for m in it:
                    row = msg_to_row(m)

                    w.writerow(row)
                    written += 1
                    last_id = m.id

                    if args.progress_every and written % args.progress_every == 0:
                        print(f"Written {written} messages (last_id={last_id})", file=sys.stderr)

                break
            except FloodWaitError as e:
                wait_s = max(int(getattr(e, "seconds", 1) or 1), 1)
                print(f"FloodWaitError: sleeping {wait_s}s...", file=sys.stderr)
                await asyncio.sleep(wait_s)

    await client.disconnect()

    print(f"Done. Wrote {written} messages to {args.out}")
    if written:
        print(f"Resume hint: --resume-from-id {last_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
