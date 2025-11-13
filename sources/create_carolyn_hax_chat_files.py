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
    "070425",
    "062725",
    "062025",
    "061325",
    "060625",
]


for date in dates:
    print("Creating new Carolyn Hax chat in sources directory.")
    if not os.path.exists(f"carolyn_hax_{date}_chat.md"):
        os.system(f"touch carolyn_hax_{date}_chat.md")
    else:
        print("Carolyn Hax chat already exists in sources directory.")
        continue

    print("Done.")

