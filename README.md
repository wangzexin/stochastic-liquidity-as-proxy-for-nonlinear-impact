# stochastic-liquidity-as-proxy-for-nonlinear-impact
Stochastic Liquidity as a Proxy for Nonlinear Price Impact

This codebase is the implementation of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4286108

Note that we start from LOBSTER data (https://lobsterdata.com/), which was originally built from Nasdaq TotalView-ITCH data. The codebase here starts from a binned sample of the LOBSTER data in 10 seconds, which is essentially taking the close price and net traded volume in each 10 seconds for each stock.