from tinkoff.invest import Client

TOKEN = "t.sWXWh2h48nFyH3cFr886QxrA9xNOHh2Sy6ULpJydAb0f_7_HbQqfUaRbQ6BmGI6cMRNT6fcC4VmRYW7NmzOseg"
TICKER = "LKOH"

with Client(TOKEN) as client:
    instruments = client.instruments.shares()

    for share in instruments.instruments:
        if share.ticker == TICKER:
            print("Название:", share.name)
            print("Тикер:", share.ticker)
            print("FIGI:", share.figi)
            print("Код валюты:", share.currency)
            print("Биржа:", share.exchange)
            break
    else:
        print("Тикер не найден.")
