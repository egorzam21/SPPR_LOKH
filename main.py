from tinkoff.invest import Client

token = "t.sWXWh2h48nFyH3cFr886QxrA9xNOHh2Sy6ULpJydAb0f_7_HbQqfUaRbQ6BmGI6cMRNT6fcC4VmRYW7NmzOseg"

with Client(token) as client:
    instruments = client.instruments.futures()
    for f in instruments.instruments:
        if "GLDRUBF" in f.ticker:
            print(f.ticker, f.figi, f.name)
