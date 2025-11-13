import os

dates=[
    "100325"
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
    "041825",
    "041125",
    "040425",
    "032825",
    "032125",
    "031425",
]


for date in dates:
    print(f"Creating new Carolyn Hax chat in sources/carolyn_hax_chats directory for {date}.")
    if not os.path.exists(f"sources/carolyn_hax_chats/carolyn_hax_{date}_chat.md"):
        os.system(f"touch sources/carolyn_hax_chats/carolyn_hax_{date}_chat.md")
    else:
        print(f"Carolyn Hax chat already exists in sources/carolyn_hax_chats directory for {date}.")
        continue

    print(f"Done creating Carolyn Hax chat in sources/carolyn_hax_chats directory for {date}.")

