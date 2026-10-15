"""
AI Stock Market Assistant (LangChain + LangGraph) - Deployable Streamlit Application
Assignment 1.1

Steps 1-10 are the SAME code as in stock_market_assistant.ipynb.
Step 11 adds the Streamlit user interface.

Run locally:  streamlit run app.py
"""


# Step 1: Import dependencies

import os
import json
import warnings
from datetime import datetime
from operator import itemgetter
from typing import TypedDict, Annotated, Literal

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain - LLM, prompts, output parsers, tools and runnables
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# LangChain - RAG components (splitter, embeddings, vector store, web search)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch

# LangGraph - graph, state reducer and memory
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# Supressing warnings for the sake of a clean output
warnings.filterwarnings("ignore")


# Step 2: Load environment variables from .env file

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Tavily is optional - if the key is present, web news is added to the RAG knowledge base
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

print("Environment variables loaded successfully.")


# Step 3: LLM integration (LangChain + Groq)

llm = ChatGroq(
       model="openai/gpt-oss-20b",
       temperature=0.2
   )


# Step 4: Define LangChain tools for stock data and news

@tool
def get_price_history(ticker: str, period: str = "6mo") -> dict:
    """
    Return historical daily price data (open, high, low, close, volume) for a stock ticker.
    Example:
    Input - ticker: AAPL, period: 6mo
    Output - Dictionary with dates and daily prices of Apple Inc.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        return {"error": f"Failed to download price data for '{ticker}': {e}"}

    if hist.empty:
        return {"error": f"No price data found for ticker '{ticker}'. Please check the symbol."}

    hist = hist.reset_index()
    return {
        "dates": hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": hist["Open"].round(2).tolist(),
        "high": hist["High"].round(2).tolist(),
        "low": hist["Low"].round(2).tolist(),
        "close": hist["Close"].round(2).tolist(),
        "volume": hist["Volume"].astype(int).tolist(),
    }


@tool
def calculate_technical_indicators(close_prices: list[float]) -> dict:
    """
    Calculate technical indicators from a list of daily closing prices (oldest first):
    last price, 1-day change, period return, SMA-20, SMA-50, RSI-14, MACD, annual volatility and trend signals.
    Example:
    Input - close_prices: [150.2, 151.0, 149.8, ...]
    Output - Dictionary of technical indicators.
    """
    s = pd.Series(close_prices, dtype="float64").dropna()
    if len(s) < 2:
        return {"error": "Not enough price data to calculate indicators."}

    def _r(x, n=2):
        return None if pd.isna(x) else round(float(x), n)

    last_price = s.iloc[-1]

    # Moving averages
    sma_20 = s.rolling(20).mean().iloc[-1]
    sma_50 = s.rolling(50).mean().iloc[-1]

    # RSI (14)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = (100 - 100 / (1 + gain / loss)).iloc[-1]

    # MACD (12, 26, 9)
    ema_12 = s.ewm(span=12, adjust=False).mean()
    ema_26 = s.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal_line = macd.ewm(span=9, adjust=False).mean()

    # Annualised volatility
    volatility = s.pct_change().std() * (252 ** 0.5) * 100

    # Trend signal
    if pd.notna(sma_20) and pd.notna(sma_50):
        if last_price > sma_20 > sma_50:
            trend = "Strong Uptrend (Price > SMA20 > SMA50)"
        elif last_price < sma_20 < sma_50:
            trend = "Strong Downtrend (Price < SMA20 < SMA50)"
        elif sma_20 > sma_50:
            trend = "Mild Uptrend (SMA20 above SMA50)"
        else:
            trend = "Mild Downtrend (SMA20 below SMA50)"
    else:
        trend = "Insufficient data for trend (need 50+ trading days)"

    if pd.isna(rsi):
        rsi_signal = "Not available"
    elif rsi > 70:
        rsi_signal = "Overbought (RSI > 70)"
    elif rsi < 30:
        rsi_signal = "Oversold (RSI < 30)"
    else:
        rsi_signal = "Neutral (RSI 30-70)"

    macd_signal = (
        "Bullish (MACD above signal line)"
        if macd.iloc[-1] > macd_signal_line.iloc[-1]
        else "Bearish (MACD below signal line)"
    )

    return {
        "last_price": _r(last_price),
        "change_1d_pct": _r((s.iloc[-1] / s.iloc[-2] - 1) * 100),
        "period_return_pct": _r((s.iloc[-1] / s.iloc[0] - 1) * 100),
        "period_high": _r(s.max()),
        "period_low": _r(s.min()),
        "sma_20": _r(sma_20),
        "sma_50": _r(sma_50),
        "rsi_14": _r(rsi),
        "rsi_signal": rsi_signal,
        "macd": _r(macd.iloc[-1], 3),
        "macd_signal_line": _r(macd_signal_line.iloc[-1], 3),
        "macd_signal": macd_signal,
        "annual_volatility_pct": _r(volatility),
        "trend": trend,
        "trading_days": int(len(s)),
    }


@tool
def get_company_fundamentals(ticker: str) -> dict:
    """
    Return key fundamentals for a stock ticker: company name, sector, market cap, P/E, EPS,
    profit margin, revenue growth, debt-to-equity, beta, 52-week range and analyst consensus.
    Example:
    Input - ticker: MSFT
    Output - Dictionary of fundamentals of Microsoft.
    """
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception as e:
        return {"error": f"Failed to fetch fundamentals for '{ticker}': {e}"}

    fields = {
        "company_name": "longName",
        "sector": "sector",
        "industry": "industry",
        "currency": "currency",
        "current_price": "currentPrice",
        "market_cap": "marketCap",
        "trailing_pe": "trailingPE",
        "forward_pe": "forwardPE",
        "eps": "trailingEps",
        "dividend_yield": "dividendYield",
        "beta": "beta",
        "52w_high": "fiftyTwoWeekHigh",
        "52w_low": "fiftyTwoWeekLow",
        "profit_margin": "profitMargins",
        "revenue_growth": "revenueGrowth",
        "debt_to_equity": "debtToEquity",
        "analyst_consensus": "recommendationKey",
        "analyst_target_price": "targetMeanPrice",
    }
    data = {key: info.get(yf_key) for key, yf_key in fields.items()}
    data["company_name"] = data["company_name"] or info.get("shortName") or ticker
    data["business_summary"] = (info.get("longBusinessSummary") or "")[:500]
    return data


@tool
def get_stock_news(ticker: str, company_name: str = "", max_articles: int = 10) -> list[dict]:
    """
    Return the latest financial news articles (title, summary, publisher, url, date) for a stock.
    Sources: Yahoo Finance news + Tavily web search (if TAVILY_API_KEY is available).
    Example:
    Input - ticker: TSLA, company_name: Tesla
    Output - List of latest news articles about Tesla.
    """
    articles = []

    # Source 1: Yahoo Finance news
    try:
        for item in (yf.Ticker(ticker).news or [])[:max_articles]:
            content = item.get("content", item)     # new and old yfinance formats
            provider = content.get("provider") or {}
            url_info = content.get("canonicalUrl") or {}
            date = content.get("pubDate") or ""
            if not date and item.get("providerPublishTime"):
                date = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d")
            article = {
                "title": content.get("title", ""),
                "summary": content.get("summary") or content.get("description") or "",
                "publisher": provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", ""),
                "url": url_info.get("url", "") if isinstance(url_info, dict) else item.get("link", ""),
                "date": str(date)[:10],
            }
            if article["title"]:
                articles.append(article)
    except Exception as e:
        print(f"Yahoo Finance news error: {e}")

    # Source 2: Tavily web search (optional)
    if TAVILY_API_KEY:
        try:
            tavily_search = TavilySearch(max_results=5, topic="news")
            response = tavily_search.invoke({"query": f"{company_name or ticker} ({ticker}) stock latest news"})
            for r in response.get("results", []):
                articles.append({
                    "title": r.get("title", ""),
                    "summary": (r.get("content") or "")[:1500],
                    "publisher": r.get("url", "").split("/")[2] if r.get("url", "").count("/") >= 2 else "Web",
                    "url": r.get("url", ""),
                    "date": str(r.get("published_date", ""))[:10],
                })
        except Exception as e:
            print(f"Tavily search error: {e}")

    return articles


# Step 5: RAG pipeline - embeddings, vector store, retriever and RAG chain

# Embedding model (runs locally, free)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Split the documents in manageable chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75
)


def build_news_retriever(articles: list[dict]):
    """Convert news articles -> Documents -> chunks -> FAISS vector store -> retriever."""
    documents = [
        Document(
            page_content=f"Title: {a['title']}\nDate: {a['date']}\nPublisher: {a['publisher']}\n\n{a['summary']}",
            metadata={"title": a["title"], "url": a["url"], "publisher": a["publisher"], "date": a["date"]},
        )
        for a in articles
    ]
    doc_chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(doc_chunks, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 6})


def format_docs(docs) -> str:
    """Join retrieved chunks into one context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Structured output schema for the news sentiment
class NewsInsight(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral", "Mixed"] = Field(description="Overall news sentiment for the stock")
    sentiment_score: float = Field(ge=-1, le=1, description="Sentiment score from -1 (very negative) to +1 (very positive)")
    key_themes: list[str] = Field(description="3-5 key themes / catalysts found in the news")
    summary: str = Field(description="4-6 sentence summary of what the news means for the stock")


news_parser = PydanticOutputParser(pydantic_object=NewsInsight)

news_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Financial News Analyst. Answer ONLY from the retrieved news context. "
     "If the context is not related to the company, say so and keep the sentiment Neutral."),
    ("user",
     "Company: {company} ({ticker})\n\n"
     "Question: {question}\n\n"
     "Retrieved news context:\n{context}\n\n"
     "{format_instructions}"),
]).partial(format_instructions=news_parser.get_format_instructions())


def build_rag_chain(retriever):
    """LCEL RAG chain: retrieve context in parallel -> prompt -> LLM -> structured output."""
    return (
        RunnableParallel(
            context=itemgetter("question") | retriever | RunnableLambda(format_docs),
            question=itemgetter("question"),
            company=itemgetter("company"),
            ticker=itemgetter("ticker"),
        )
        | news_prompt
        | llm
        | news_parser
    )


# Step 6: Pydantic schemas + prompt templates + chains for the agents

# ---------- Query Router ----------
class QueryPlan(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol e.g. AAPL, MSFT, TSLA, RELIANCE.NS, TCS.NS. Empty string if no stock/company is mentioned.")
    company_name: str = Field(description="Company name. Empty string if no stock/company is mentioned.")
    intent: Literal["full_analysis", "price_check", "news_only"] = Field(description="Type of request")
    period: Literal["3mo", "6mo", "1y", "2y", "5y"] = Field(description="Historical period to analyse")


router_parser = PydanticOutputParser(pydantic_object=QueryPlan)

router_prompt = PromptTemplate(
    template='''
    You are the Query Router of an AI Stock Market Assistant.
    Read the user query and extract the stock details.

    Rules:
    1. Convert the company name into the correct Yahoo Finance ticker.
       - US stocks: plain symbol (Apple -> AAPL, Microsoft -> MSFT, Nvidia -> NVDA, Tesla -> TSLA).
       - Indian NSE stocks: add ".NS" (Reliance -> RELIANCE.NS, TCS -> TCS.NS, Infosys -> INFY.NS, HDFC Bank -> HDFCBANK.NS).
       - If the user already gives a ticker, keep it as it is.
    2. intent:
       - "price_check"  -> user only asks about price, performance, trend or technical indicators.
       - "news_only"    -> user only asks about news / latest updates.
       - "full_analysis"-> everything else, including "should I buy/sell/hold", "analyse", "invest", "recommendation".
    3. period: use "6mo" by default. Use "3mo" for 1-3 months, "1y" for one year, "2y" or "5y" for longer periods.
    4. If the query is not about any stock or company, return ticker as an empty string.

    User query: {query}

    {format_instructions}
    ''',
    input_variables=["query"],
    partial_variables={"format_instructions": router_parser.get_format_instructions()},
)

router_chain = router_prompt | llm | router_parser


# ---------- Analyst Agent ----------
analyst_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Senior Equity Research Analyst with 15+ years of experience. "
     "You explain stock data clearly to retail investors. Use ONLY the data provided - never invent numbers."),
    ("user",
     '''User question: {query}
Company: {company} ({ticker}) | Request type: {intent} | Period: {period}

### Technical indicators
{indicators}

### Company fundamentals
{fundamentals}

### News sentiment (generated by RAG over the latest news)
{news}

Write a concise, well-structured analysis in markdown with these sections:
1. **Price & Trend Overview**
2. **Technical Signals** - interpret SMA, RSI, MACD and volatility
3. **Fundamental Snapshot** - valuation, profitability, growth, debt
4. **News & Sentiment Impact**
5. **Key Insights** - 3 to 5 bullet points

If the data for a section says "Not requested" or "Not available", write "Not available for this request" for that section.
Do not give a Buy/Sell/Hold recommendation here. Mention the currency where relevant.'''),
])

analyst_chain = analyst_prompt | llm | StrOutputParser()


# ---------- Recommendation Agent ----------
class Recommendation(BaseModel):
    decision: Literal["BUY", "SELL", "HOLD"] = Field(description="Simulated recommendation")
    confidence: int = Field(ge=0, le=100, description="Confidence in the decision from 0 to 100")
    time_horizon: str = Field(description="Suggested time horizon e.g. 'Short term (1-3 months)'")
    justification: list[str] = Field(description="3-5 bullet point reasons based on the data")
    key_risks: list[str] = Field(description="2-4 key risks to watch")


recommendation_parser = PydanticOutputParser(pydantic_object=Recommendation)

recommendation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Portfolio Strategist. You give SIMULATED, educational Buy/Sell/Hold recommendations "
     "based strictly on the analysis provided. This is not real financial advice."),
    ("user",
     '''Company: {company} ({ticker})

### Analyst report
{analysis}

### Technical indicators
{indicators}

### News sentiment
{news}

### Valuation & analyst consensus
{fundamentals}

Decision guidelines:
- BUY  -> trend and momentum are positive, news sentiment is positive/neutral and valuation is reasonable.
- SELL -> trend and momentum are negative and news sentiment is negative, or there is a serious risk.
- HOLD -> signals are mixed or there is not enough evidence.
Confidence should reflect how strongly the signals agree with each other.

{format_instructions}'''),
]).partial(format_instructions=recommendation_parser.get_format_instructions())

recommendation_chain = recommendation_prompt | llm | recommendation_parser


# Step 7: Define the state schema shared by all agents

class StockState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # conversation history (reducer appends)
    user_query: str
    ticker: str
    company_name: str
    intent: str              # full_analysis | price_check | news_only
    period: str
    error: str
    market_data: dict        # price history + technical indicators + fundamentals
    news_articles: list      # raw news articles
    news_insight: dict       # RAG output - sentiment, themes, summary
    analysis: str            # Analyst agent output
    recommendation: dict     # Recommendation agent output
    final_report: str        # Final markdown report
    agent_log: list          # Trace of which agents ran


# Step 8: Define the agent nodes

def to_json(data) -> str:
    """Helper - convert dict to readable JSON for prompts."""
    return json.dumps(data, indent=2, default=str)


# Agent 1: Query Router
def query_router_agent(state: StockState):
    log = state.get("agent_log", []) + ["🧭 Query Router Agent: understanding the request"]
    try:
        plan = router_chain.invoke({"query": state["user_query"]})
    except Exception as e:
        return {"error": f"Could not understand the query: {e}", "agent_log": log}

    log.append(f"   → ticker = {plan.ticker or 'N/A'}, intent = {plan.intent}, period = {plan.period}")
    return {
        "ticker": plan.ticker.strip().upper(),
        "company_name": plan.company_name,
        "intent": plan.intent,
        "period": plan.period,
        "agent_log": log,
    }


# Agent 2: Market Data
def market_data_agent(state: StockState):
    log = state["agent_log"] + [f"📊 Market Data Agent: fetching {state['period']} price history, indicators and fundamentals"]
    ticker = state["ticker"]

    history = get_price_history.invoke({"ticker": ticker, "period": state["period"]})
    if "error" in history:
        return {"error": history["error"], "agent_log": log}

    indicators = calculate_technical_indicators.invoke({"close_prices": history["close"]})
    fundamentals = get_company_fundamentals.invoke({"ticker": ticker})

    log.append(f"   → {indicators.get('trading_days')} trading days, last price {indicators.get('last_price')}, trend: {indicators.get('trend')}")
    return {
        "market_data": {"history": history, "indicators": indicators, "fundamentals": fundamentals},
        "company_name": fundamentals.get("company_name") or state["company_name"],
        "agent_log": log,
    }


# Agent 3: News (RAG)
def news_agent(state: StockState):
    log = state["agent_log"] + ["📰 News Agent (RAG): retrieving and analysing the latest news"]
    ticker, company = state["ticker"], state["company_name"] or state["ticker"]

    articles = get_stock_news.invoke({"ticker": ticker, "company_name": company})

    if not articles:
        insight = {"sentiment": "Neutral", "sentiment_score": 0.0, "key_themes": [],
                   "summary": "No recent news articles were found for this stock."}
        log.append("   → no news found")
    else:
        try:
            retriever = build_news_retriever(articles)
            rag_chain = build_rag_chain(retriever)
            result = rag_chain.invoke({
                "question": f"What are the latest developments, risks and growth catalysts for {company} ({ticker}) stock?",
                "company": company,
                "ticker": ticker,
            })
            insight = result.model_dump()
            log.append(f"   → {len(articles)} articles indexed in FAISS, sentiment = {insight['sentiment']} ({insight['sentiment_score']})")
        except Exception as e:
            insight = {"sentiment": "Neutral", "sentiment_score": 0.0, "key_themes": [],
                       "summary": f"News analysis failed: {e}"}
            log.append(f"   → RAG error: {type(e).__name__}: {e}")

    return {"news_articles": articles, "news_insight": insight, "agent_log": log}


# Agent 4: Analyst
def analyst_agent(state: StockState):
    log = state["agent_log"] + ["🧠 Analyst Agent: generating insights with LLM reasoning"]
    market = state.get("market_data") or {}

    analysis = analyst_chain.invoke({
        "query": state["user_query"],
        "company": state["company_name"],
        "ticker": state["ticker"],
        "intent": state["intent"],
        "period": state["period"],
        "indicators": to_json(market.get("indicators")) if market else "Not requested",
        "fundamentals": to_json(market.get("fundamentals")) if market else "Not requested",
        "news": to_json(state["news_insight"]) if state.get("news_insight") else "Not requested",
    })
    return {"analysis": analysis, "agent_log": log}


# Agent 5: Recommendation
def recommendation_agent(state: StockState):
    log = state["agent_log"] + ["🎯 Recommendation Agent: deciding BUY / SELL / HOLD"]
    market = state.get("market_data") or {}
    fundamentals = market.get("fundamentals", {})

    try:
        rec = recommendation_chain.invoke({
            "company": state["company_name"],
            "ticker": state["ticker"],
            "analysis": state["analysis"],
            "indicators": to_json(market.get("indicators", {})),
            "news": to_json(state.get("news_insight", {})),
            "fundamentals": to_json({k: fundamentals.get(k) for k in
                                     ["trailing_pe", "forward_pe", "profit_margin", "revenue_growth",
                                      "debt_to_equity", "analyst_consensus", "analyst_target_price", "currency"]}),
        }).model_dump()
    except Exception as e:
        rec = {"decision": "HOLD", "confidence": 0, "time_horizon": "N/A",
               "justification": [f"Could not generate a structured recommendation: {e}"], "key_risks": []}

    log.append(f"   → {rec['decision']} (confidence {rec['confidence']}%)")
    return {"recommendation": rec, "agent_log": log}


# Agent 6: Report Generator
def report_generator(state: StockState):
    log = state["agent_log"] + ["📝 Report Generator: compiling the final report"]
    intent_label = {"full_analysis": "Full Analysis", "price_check": "Price & Technical Check",
                    "news_only": "News Summary"}.get(state["intent"], state["intent"])

    report = f"# 📈 {state['company_name']} ({state['ticker']}) – Stock Analysis Report\n"
    report += f"*Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} | Request: {intent_label} | Period: {state['period']}*\n\n"

    rec = state.get("recommendation")
    if rec:
        report += f"## 🎯 Simulated Recommendation: **{rec['decision']}** (Confidence: {rec['confidence']}%)\n"
        report += f"**Time horizon:** {rec['time_horizon']}\n\n**Justification:**\n"
        report += "".join(f"- {reason}\n" for reason in rec["justification"])
        report += "\n**Key risks:**\n" + "".join(f"- {risk}\n" for risk in rec["key_risks"]) + "\n"

    report += f"## 🧠 Analyst Insights\n{state['analysis']}\n\n"

    if state.get("news_articles"):
        report += "## 📰 Top News Headlines\n"
        for i, a in enumerate(state["news_articles"][:6], start=1):
            title = f"[{a['title']}]({a['url']})" if a["url"] else a["title"]
            report += f"{i}. {title} – *{a['publisher']}* {a['date']}\n"
        report += "\n"

    report += ("---\n> ⚠️ **Disclaimer:** This is a simulated recommendation generated by an AI system for "
               "educational purposes only. It is not financial advice.")

    return {"final_report": report, "messages": [AIMessage(content=report)], "agent_log": log}


# Error handler
def invalid_request(state: StockState):
    log = state.get("agent_log", []) + ["⚠️ Invalid Request handler"]
    reason = state.get("error") or "I could not find a stock or company in your question."
    report = (f"### ⚠️ Unable to process the request\n{reason}\n\n"
              "Please ask about a listed stock, for example:\n"
              "- *Should I buy Apple stock?*\n"
              "- *How has TCS performed in the last 3 months?*\n"
              "- *Latest news about Tesla*")
    return {"final_report": report, "messages": [AIMessage(content=report)], "agent_log": log}


# Step 9: Routing functions for the conditional edges

def route_after_router(state: StockState) -> Literal["market_data_agent", "news_agent", "invalid_request"]:
    if state.get("error") or not state.get("ticker"):
        return "invalid_request"
    if state["intent"] == "news_only":
        return "news_agent"
    return "market_data_agent"


def route_after_market_data(state: StockState) -> Literal["news_agent", "analyst_agent", "invalid_request"]:
    if state.get("error"):
        return "invalid_request"
    if state["intent"] == "price_check":
        return "analyst_agent"
    return "news_agent"


def route_after_analyst(state: StockState) -> Literal["recommendation_agent", "report_generator"]:
    if state["intent"] == "full_analysis":
        return "recommendation_agent"
    return "report_generator"


# Step 10: Build the LangGraph workflow - nodes, edges, conditional edges and memory

def build_stock_graph():
    builder = StateGraph(StockState)

    # Add nodes (agents)
    builder.add_node("query_router_agent", query_router_agent)
    builder.add_node("market_data_agent", market_data_agent)
    builder.add_node("news_agent", news_agent)
    builder.add_node("analyst_agent", analyst_agent)
    builder.add_node("recommendation_agent", recommendation_agent)
    builder.add_node("report_generator", report_generator)
    builder.add_node("invalid_request", invalid_request)

    # Normal edges
    builder.add_edge(START, "query_router_agent")
    builder.add_edge("news_agent", "analyst_agent")
    builder.add_edge("recommendation_agent", "report_generator")
    builder.add_edge("report_generator", END)
    builder.add_edge("invalid_request", END)

    # Conditional edges
    builder.add_conditional_edges("query_router_agent", route_after_router)
    builder.add_conditional_edges("market_data_agent", route_after_market_data)
    builder.add_conditional_edges("analyst_agent", route_after_analyst)

    # Short-term memory (checkpointer) - keeps conversation per thread_id
    return builder.compile(checkpointer=InMemorySaver())


stock_graph = build_stock_graph()
print("Multi-agent workflow compiled successfully.")


def make_initial_state(user_query: str) -> dict:
    """Fresh input state for a new query (old values are reset, messages are appended to memory)."""
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "ticker": "",
        "company_name": "",
        "intent": "",
        "period": "6mo",
        "error": "",
        "market_data": {},
        "news_articles": [],
        "news_insight": {},
        "analysis": "",
        "recommendation": {},
        "final_report": "",
        "agent_log": [],
    }


def run_stock_assistant(user_query: str, thread_id: str = "demo-thread") -> dict:
    """Run the multi-agent workflow for one user query and return the final state."""
    config = {"configurable": {"thread_id": thread_id}}
    return stock_graph.invoke(make_initial_state(user_query), config=config)


def history_to_dataframe(history: dict) -> pd.DataFrame:
    """Helper - convert price history into a DataFrame with SMA lines for charts."""
    df = pd.DataFrame({"Close": history["close"]}, index=pd.to_datetime(history["dates"]))
    df["SMA 20"] = df["Close"].rolling(20).mean()
    df["SMA 50"] = df["Close"].rolling(50).mean()
    return df


# Step 11: Streamlit UI - Deployable application
# Run with: streamlit run app.py

import uuid
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AI Stock Market Assistant", page_icon="📈", layout="wide")

# ---------- Session state ----------
for key, value in {"thread_id": str(uuid.uuid4()), "last_result": None, "history": []}.items():
    if key not in st.session_state:
        st.session_state[key] = value

AGENT_LABELS = {
    "query_router_agent": "🧭 **Query Router** – understood the question",
    "market_data_agent": "📊 **Market Data Agent** – fetched prices, indicators and fundamentals",
    "news_agent": "📰 **News Agent (RAG)** – indexed the latest news and analysed sentiment",
    "analyst_agent": "🧠 **Analyst Agent** – generated insights",
    "recommendation_agent": "🎯 **Recommendation Agent** – decided Buy / Sell / Hold",
    "report_generator": "📝 **Report Generator** – compiled the report",
    "invalid_request": "⚠️ **Invalid Request** handler",
}

EXAMPLE_QUERIES = [
    "Should I buy Apple stock?",
    "Analyse Reliance Industries",
    "TCS performance last 3 months",
    "Latest news about Tesla",
    "Is NVDA a buy, sell or hold?",
]

# Chart range -> number of trading days (None = full history)
CHART_RANGES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "Max": None}


# ---------- Helper functions ----------
def run_with_live_progress(query: str) -> dict:
    """Stream the LangGraph workflow so the user sees each agent finishing in real time."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.status(f"Agents are working on: *{query}*", expanded=True) as status:
        for update in stock_graph.stream(make_initial_state(query), config=config, stream_mode="updates"):
            for node in update:
                st.write("✅ " + AGENT_LABELS.get(node, node))
        result = stock_graph.get_state(config).values
        status.update(label="Analysis complete", state="complete", expanded=False)
    return result


def open_from_history(index: int):
    st.session_state.last_result = st.session_state.history[index]["result"]


def clear_history():
    st.session_state.history = []
    st.session_state.last_result = None


@st.cache_data(ttl=900, show_spinner=False)
def load_chart_data(ticker: str, period: str) -> pd.DataFrame:
    """Download long price history once (cached 15 min) and pre-compute moving averages."""
    history = get_price_history.invoke({"ticker": ticker, "period": period})
    if "error" in history:
        return pd.DataFrame()
    df = pd.DataFrame(
        {col.capitalize(): history[col] for col in ["open", "high", "low", "close", "volume"]},
        index=pd.to_datetime(history["dates"]),
    )
    for window in (20, 50, 200):
        df[f"SMA {window}"] = df["Close"].rolling(window).mean()
    return df


def build_price_chart(df: pd.DataFrame, chart_type: str, overlays: list, title: str) -> go.Figure:
    show_volume = "Volume" in overlays
    fig = make_subplots(
        rows=2 if show_volume else 1, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22] if show_volume else [1.0], vertical_spacing=0.03,
    )
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                     close=df["Close"], name="Price"), row=1, col=1)
    else:
        trend_up = df["Close"].iloc[-1] >= df["Close"].iloc[0]
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                 line=dict(color="#22c55e" if trend_up else "#ef4444", width=2)), row=1, col=1)

    sma_colors = {"SMA 20": "#f59e0b", "SMA 50": "#3b82f6", "SMA 200": "#a855f7"}
    for sma, color in sma_colors.items():
        if sma in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=df[sma], name=sma,
                                     line=dict(color=color, width=1.4)), row=1, col=1)

    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color="rgba(148,163,184,0.5)"), row=2, col=1)

    fig.update_layout(
        title=title, height=560, hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def range_statistics(df: pd.DataFrame) -> dict:
    close = df["Close"]
    years = (df.index[-1] - df.index[0]).days / 365.25
    growth = close.iloc[-1] / close.iloc[0]
    return {
        "return": (growth - 1) * 100,
        "cagr": (growth ** (1 / years) - 1) * 100 if years >= 0.95 else None,
        "high": close.max(),
        "low": close.min(),
        "drawdown": ((close / close.cummax()) - 1).min() * 100,
    }


def render_price_chart_tab(ticker: str, company: str, currency: str):
    c1, c2, c3 = st.columns([3, 1.6, 3])
    with c1:
        range_label = st.segmented_control("Time range", list(CHART_RANGES), default="1Y", key="chart_range") or "1Y"
    with c2:
        chart_type = st.segmented_control("Chart type", ["Line", "Candlestick"], default="Line", key="chart_type") or "Line"
    with c3:
        overlays = st.pills("Overlays", ["SMA 20", "SMA 50", "SMA 200", "Volume"],
                            selection_mode="multi", default=["SMA 50", "SMA 200", "Volume"], key="chart_overlays") or []

    with st.spinner("Loading price history..."):
        full_df = load_chart_data(ticker, "max" if range_label == "Max" else "5y")
    if full_df.empty:
        st.warning(f"No price history available for {ticker}.")
        return

    days = CHART_RANGES[range_label]
    df = full_df.tail(days) if days else full_df
    stats = range_statistics(df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Return ({range_label})", f"{stats['return']:.2f}%")
    m2.metric("CAGR", f"{stats['cagr']:.2f}%" if stats["cagr"] is not None else "–",
              help="Compound annual growth rate (shown for ranges of 1 year or more)")
    suffix = f" ({currency})" if currency else ""
    m3.metric("High" + suffix, f"{stats['high']:,.2f}")
    m4.metric("Low" + suffix, f"{stats['low']:,.2f}")
    m5.metric("Max drawdown", f"{stats['drawdown']:.2f}%", help="Largest fall from a previous peak in this range")

    if days and len(df) < days:
        st.caption(f"Only {len(df)} trading days of history are available for {ticker}.")

    st.plotly_chart(
        build_price_chart(df, chart_type, overlays, f"{company} ({ticker}) – {range_label} trend"),
        width="stretch",
    )
    st.download_button("⬇️ Download price data (.csv)", df.to_csv().encode("utf-8"),
                       file_name=f"{ticker}_{range_label}_prices.csv", mime="text/csv")


def render_result(result: dict):
    market = result.get("market_data") or {}
    indicators = market.get("indicators", {})
    fundamentals = market.get("fundamentals", {})
    news_insight = result.get("news_insight") or {}
    rec = result.get("recommendation") or {}
    currency = fundamentals.get("currency") or ""
    intent_label = {"full_analysis": "Full analysis", "price_check": "Price & technical check",
                    "news_only": "News summary"}.get(result["intent"], result["intent"])

    # Company header
    st.markdown(f"## {result['company_name']} ({result['ticker']})")
    details = [fundamentals.get("sector"), fundamentals.get("industry"), currency, intent_label, f"Query: “{result['user_query']}”"]
    st.caption(" · ".join(d for d in details if d))

    # Key metrics (only the ones available for this request)
    metrics = []
    if indicators:
        metrics.append((f"Last Price ({currency})" if currency else "Last Price", indicators.get("last_price"),
                        f"{indicators['change_1d_pct']}%" if indicators.get("change_1d_pct") is not None else None))
        metrics.append((f"Return ({result['period']})",
                        f"{indicators['period_return_pct']}%" if indicators.get("period_return_pct") is not None else "–", None))
        metrics.append(("RSI (14)", indicators.get("rsi_14") if indicators.get("rsi_14") is not None else "–", None))
    if news_insight:
        metrics.append(("News Sentiment", news_insight.get("sentiment", "–"), news_insight.get("sentiment_score")))
    if fundamentals.get("trailing_pe"):
        metrics.append(("P/E (TTM)", round(fundamentals["trailing_pe"], 2), None))
    if metrics:
        for col, (label, value, delta) in zip(st.columns(len(metrics)), metrics):
            col.metric(label, value, delta)

    # Recommendation card
    if rec:
        with st.container(border=True):
            message = f"### 🎯 Simulated Recommendation: {rec['decision']}  ·  Confidence {rec['confidence']}%"
            {"BUY": st.success, "SELL": st.error}.get(rec["decision"], st.warning)(message)
            st.progress(rec["confidence"] / 100, text=f"Time horizon: {rec['time_horizon']}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**✅ Justification**")
                st.markdown("\n".join(f"- {r}" for r in rec["justification"]))
            with col_b:
                st.markdown("**⚠️ Key risks**")
                st.markdown("\n".join(f"- {r}" for r in rec["key_risks"]))

    tab_chart, tab_data, tab_news, tab_analysis, tab_report, tab_trace = st.tabs(
        ["📈 Price Chart", "📊 Indicators & Fundamentals", "📰 News (RAG)", "🧠 Analysis", "📝 Report", "🔀 Agent Trace"]
    )

    with tab_chart:
        render_price_chart_tab(result["ticker"], result["company_name"], currency)

    with tab_data:
        if market:
            col_t, col_f = st.columns(2)
            with col_t:
                st.markdown(f"**Technical indicators ({result['period']})**")
                st.dataframe(pd.DataFrame(indicators.items(), columns=["Indicator", "Value"]).astype(str),
                             hide_index=True, width="stretch")
            with col_f:
                st.markdown("**Fundamentals**")
                st.dataframe(pd.DataFrame([(k, v) for k, v in fundamentals.items() if k != "business_summary"],
                                          columns=["Metric", "Value"]).astype(str),
                             hide_index=True, width="stretch")
            if fundamentals.get("business_summary"):
                with st.expander("About the company"):
                    st.write(fundamentals["business_summary"])
        else:
            st.info("Market data was not part of this request (news only). Try a full analysis for indicators.")

    with tab_news:
        if news_insight:
            st.markdown(f"**Sentiment:** {news_insight.get('sentiment')} (score {news_insight.get('sentiment_score')})")
            if news_insight.get("key_themes"):
                st.pills("Key themes", news_insight["key_themes"], key="themes_display", disabled=True)
            st.write(news_insight.get("summary", ""))
            st.markdown(f"**{len(result.get('news_articles', []))} articles indexed in the FAISS vector store**")
            for a in result.get("news_articles", []):
                with st.expander(f"{a['title']}  ·  {a['publisher']} {a['date']}"):
                    st.write(a["summary"][:800] or "No summary available.")
                    if a["url"]:
                        st.markdown(f"[Read full article ↗]({a['url']})")
        else:
            st.info("News was not part of this request (price check).")

    with tab_analysis:
        st.markdown(result["analysis"])

    with tab_report:
        st.download_button("⬇️ Download report (.md)", data=result["final_report"],
                           file_name=f"{result['ticker']}_report.md", mime="text/markdown")
        st.markdown(result["final_report"])

    with tab_trace:
        st.code("\n".join(result["agent_log"]), language="text")


# ---------- Search bar + quick picks ----------
with st.form("query_form", border=False):
    col_input, col_button = st.columns([6, 1], vertical_alignment="bottom")
    with col_input:
        user_query = st.text_input("Ask about any stock",
                                   placeholder="e.g. Should I buy Apple stock?  |  Analyse INFY.NS for 1 year  |  Latest news about Tesla")
    with col_button:
        submitted = st.form_submit_button("🔍 Analyse", type="primary", width="stretch")

clicked_example = None
for col, example in zip(st.columns(len(EXAMPLE_QUERIES)), EXAMPLE_QUERIES):
    if col.button(example, key=f"example_{example}", width="stretch"):
        clicked_example = example

query_to_run = user_query.strip() if submitted else clicked_example

if submitted and not query_to_run:
    st.error("Please enter a question about a stock.")
elif query_to_run:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
    else:
        try:
            result = run_with_live_progress(query_to_run)
            st.session_state.last_result = result
            label = f"{result['ticker'] or '–'} · {query_to_run[:32]}"
            st.session_state.history.insert(0, {"label": label, "result": result})
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()

# ---------- Results ----------
result = st.session_state.last_result
if result is None:
    st.info("👋 Ask a question about any listed stock, or click one of the quick examples above. "
            "US tickers (AAPL, MSFT) and Indian NSE stocks (RELIANCE.NS, TCS.NS) are supported.")
elif not result.get("analysis"):
    st.warning(result["final_report"])
else:
    render_result(result)

# Sidebar is drawn last so 'Recent analyses' already includes the query that just ran
# ---------- Sidebar ----------
with st.sidebar:
    st.title("📈 Stock Assistant")
    st.caption("Multi-agent system · LangChain + LangGraph")

    st.subheader("🕘 Recent analyses")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history[:8]):
            st.button(item["label"], key=f"history_{i}", on_click=open_from_history, args=(i,), width="stretch")
        st.button("🗑️ Clear history", on_click=clear_history, type="tertiary")
    else:
        st.caption("Your analyses will appear here.")

    st.caption(f"Groq API key: {'✅' if os.getenv('GROQ_API_KEY') else '❌ missing'}  ·  "
               f"Tavily: {'✅' if TAVILY_API_KEY else '➖ optional'}")
    st.warning("Simulated recommendations for educational purposes only. Not financial advice.")
