# from ib_async import *
# from ib_async.client import *
# from ib_async.wrapper import *
# from ib_async.contract import *
# # util.startLoop()  # uncomment this line when in a notebook

# ib = IB()
# ib.connect(port = 4001, clientId=0, readonly = True)
# # ib.client.connect("127.0.0.1", port = 4002, clientId=0)

# # ib.reqMarketDataType(4)  # Use free, delayed, frozen data
# contract = Stock(symbol = 'RNO',  exchange= 'SMART', currency= "EUR")

# # bars = ib.reqHistoricalData(
# #     contract, endDateTime='', durationStr='30 D',
# #     barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)
# ib.wrapper.fundamentalData(9001, 'gfgdhdfh')
# # ib.client.reqAccountSummary()
# bars = ib.reqFundamentalData(contract, reportType='RESC')


from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def fundamentalData(self, reqId, data):
        print("FundamentalData.ReqId:", reqId, "Data:", data)

def main():
    app = IBapi()
    app.connect('127.0.0.1', 4001, 0) # Connect to TWS on port 7497 with client ID 123
    contract = Contract()
    # contract = Contract(symbol = 'RNO', exchange= 'SMART', currency= "EUR")
    contract.symbol = "RNO"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "EUR"
    # contract.primaryExchange = "NASDAQ"
    app.reqFundamentalData(1, contract, "RESC", []) # Request financial statements
    app.run()

if __name__ == "__main__":
    main()