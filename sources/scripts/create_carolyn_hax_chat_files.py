from pathlib import Path

dates = [
    "100325",
    "092625",
    "091925",
    "091225",
    "090525",
    "072525",
    "071825",
    "071125",
    # "070425", # This chat is missing.
    "062725",
    # "062025", # This chat is missing.
    "061325",
    "060625",
    "051625",
    "050925",
    "050225",
    "042525",
    # "041825", # This chat is missing.
    "041125",
    "040425",
    "032825",
    "032125",
    "031425",
]

# Determine paths relative to script location
script_dir = Path(__file__).parent
sources_dir = script_dir.parent  # Go up from scripts/ to sources/
chats_dir = sources_dir / "carolyn_hax" / "carolyn_hax_chats"

for date in dates:
    chat_file = chats_dir / f"carolyn_hax_{date}_chat.md"
    print(f"Creating new Carolyn Hax chat in {chats_dir} directory for {date}.")
    if not chat_file.exists():
        chat_file.parent.mkdir(parents=True, exist_ok=True)
        chat_file.touch()
    else:
        print(f"Carolyn Hax chat already exists in {chats_dir} directory for {date}.")
        continue

    print(f"Done creating Carolyn Hax chat in {chats_dir} directory for {date}.")
