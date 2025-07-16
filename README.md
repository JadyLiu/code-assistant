# Code Assistant Demo

Imagine having an intelligent assistant that boosts your productivity as a developer. This assistant understands and generates code, making developer's daily tasks smoother and more efficient. It can automatically turn issues into pull requests, create useful code snippets, and offer suggestions tailored to your coding context. By understanding your existing codebase, it helps developers to write, review, and manage code with ease. This not only cuts down on manual coding chores but also speeds up the entire development process.

## Demo Scenarios

demonstrating AI capabilities in:

1. **Code Analysis**: "Explain how the RSI strategy works"
2. **Feature Addition**: "Add a Bollinger Bands strategy"
3. **Risk Enhancement**: "Implement stop-loss functionality"
4. **Performance Optimization**: "Optimize strategy parameters"
5. **Testing**: "Write tests for the new features"
6. **Documentation**: "Generate API documentation"

1.How do I add a new trading strategy to this system? - Shows code extension understanding
2.What's the risk management approach and how can I modify position sizing? - Demonstrates domain-specific knowledge extraction
3.Walk me through the backtesting workflow from data to results - Shows end-to-end process comprehension
4.What are the key classes and their relationships in this trading system? - Architecture understanding
5.How does the portfolio handle edge cases like insufficient funds or positions? - Error handling and defensive programming insights


Demo Prompts

"Generate a new trading indicator strategy following our coding style standards"
"Refactor this poorly written function to match our style guide"
"Create unit tests following our existing test patterns"

```python
import math

class badIndicator:
    def __init__(self,data):
        self.data=data
        self.results=[]
    
    def calcRSI(self,period=14):
        # calculate RSI without proper validation or documentation
        gains=[]
        losses=[]
        for i in range(1,len(self.data)):
            change=self.data[i]-self.data[i-1]
            if change>0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # hardcoded magic numbers everywhere
        avgGain=sum(gains[:14])/14
        avgLoss=sum(losses[:14])/14
        
        rsi_values=[]
        for i in range(period,len(gains)):
            if avgLoss==0:
                rsi=100
            else:
                rs=avgGain/avgLoss
                rsi=100-(100/(1+rs))
            rsi_values.append(rsi)
            
            # recalculate averages - inefficient
            avgGain=(avgGain*13+gains[i])/14
            avgLoss=(avgLoss*13+losses[i])/14
        
        return rsi_values
    
    def GetSignal(self,rsi_val):
        # inconsistent naming, no type hints
        if rsi_val>70:
            return "SELL"
        elif rsi_val<30:
            return "BUY"
        else:
            return "HOLD"
    
    def backtest_strategy(self,initial_cash,prices):
        # no error handling, mixed responsibilities
        cash=initial_cash
        position=0
        trades=[]
        
        rsi_values=self.calcRSI()
        
        for i in range(len(rsi_values)):
            signal=self.GetSignal(rsi_values[i])
            price=prices[i+14] # magic number offset
            
            if signal=="BUY" and cash>price:
                shares=int(cash/price) # buy maximum shares
                cash-=shares*price
                position+=shares
                trades.append(["BUY",shares,price])
            elif signal=="SELL" and position>0:
                cash+=position*price
                trades.append(["SELL",position,price])
                position=0
        
        # return inconsistent data structure
        return {"cash":cash,"position":position,"trades":trades,"final_value":cash+position*prices[-1]}

# Global variables - bad practice
DEFAULT_PERIOD=14
RSI_OVERBOUGHT=70
RSI_OVERSOLD=30

def quick_rsi_calc(data):
    # function without class, inconsistent with codebase style
    indicator=badIndicator(data)
    return indicator.calcRSI(DEFAULT_PERIOD)

# No main guard, hardcoded test
test_data=[100,102,101,103,105,104,106,108,107,109,111,110,112,114,113]
rsi_calc=badIndicator(test_data)
print("RSI values:",rsi_calc.calcRSI())|
```


python -m unittest discover -s tests